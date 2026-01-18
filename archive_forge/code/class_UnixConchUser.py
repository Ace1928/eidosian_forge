from __future__ import annotations
import fcntl
import grp
import os
import pty
import pwd
import socket
import struct
import time
import tty
from typing import Callable, Dict, Tuple
from zope.interface import implementer
from twisted.conch import ttymodes
from twisted.conch.avatar import ConchUser
from twisted.conch.error import ConchError
from twisted.conch.interfaces import ISession, ISFTPFile, ISFTPServer
from twisted.conch.ls import lsLine
from twisted.conch.ssh import filetransfer, forwarding, session
from twisted.conch.ssh.filetransfer import (
from twisted.cred import portal
from twisted.cred.error import LoginDenied
from twisted.internet.error import ProcessExitedAlready
from twisted.internet.interfaces import IListeningPort
from twisted.logger import Logger
from twisted.python import components
from twisted.python.compat import nativeString
class UnixConchUser(ConchUser):

    def __init__(self, username: str) -> None:
        ConchUser.__init__(self)
        self.username = username
        self.pwdData = pwd.getpwnam(self.username)
        l = [self.pwdData[3]]
        for groupname, password, gid, userlist in grp.getgrall():
            if username in userlist:
                l.append(gid)
        self.otherGroups = l
        self.listeners: Dict[str, IListeningPort] = {}
        self.channelLookup.update({b'session': session.SSHSession, b'direct-tcpip': forwarding.openConnectForwardingClient})
        self.subsystemLookup.update({b'sftp': filetransfer.FileTransferServer})

    def getUserGroupId(self):
        return self.pwdData[2:4]

    def getOtherGroups(self):
        return self.otherGroups

    def getHomeDir(self):
        return self.pwdData[5]

    def getShell(self):
        return self.pwdData[6]

    def global_tcpip_forward(self, data):
        hostToBind, portToBind = forwarding.unpackGlobal_tcpip_forward(data)
        from twisted.internet import reactor
        try:
            listener = self._runAsUser(reactor.listenTCP, portToBind, forwarding.SSHListenForwardingFactory(self.conn, (hostToBind, portToBind), forwarding.SSHListenServerForwardingChannel), interface=hostToBind)
        except BaseException:
            return 0
        else:
            self.listeners[hostToBind, portToBind] = listener
            if portToBind == 0:
                portToBind = listener.getHost()[2]
                return (1, struct.pack('>L', portToBind))
            else:
                return 1

    def global_cancel_tcpip_forward(self, data):
        hostToBind, portToBind = forwarding.unpackGlobal_tcpip_forward(data)
        listener = self.listeners.get((hostToBind, portToBind), None)
        if not listener:
            return 0
        del self.listeners[hostToBind, portToBind]
        self._runAsUser(listener.stopListening)
        return 1

    def logout(self) -> None:
        for listener in self.listeners.values():
            self._runAsUser(listener.stopListening)
        self._log.info('avatar {username} logging out ({nlisteners})', username=self.username, nlisteners=len(self.listeners))

    def _runAsUser(self, f, *args, **kw):
        euid = os.geteuid()
        egid = os.getegid()
        groups = os.getgroups()
        uid, gid = self.getUserGroupId()
        os.setegid(0)
        os.seteuid(0)
        os.setgroups(self.getOtherGroups())
        os.setegid(gid)
        os.seteuid(uid)
        try:
            f = iter(f)
        except TypeError:
            f = [(f, args, kw)]
        try:
            for i in f:
                func = i[0]
                args = len(i) > 1 and i[1] or ()
                kw = len(i) > 2 and i[2] or {}
                r = func(*args, **kw)
        finally:
            os.setegid(0)
            os.seteuid(0)
            os.setgroups(groups)
            os.setegid(egid)
            os.seteuid(euid)
        return r