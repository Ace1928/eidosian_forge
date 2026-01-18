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
@implementer(ISession)
class SSHSessionForUnixConchUser:
    _log = Logger()

    def __init__(self, avatar, reactor=None):
        """
        Construct an C{SSHSessionForUnixConchUser}.

        @param avatar: The L{UnixConchUser} for whom this is an SSH session.
        @param reactor: An L{IReactorProcess} used to handle shell and exec
            requests. Uses the default reactor if None.
        """
        if reactor is None:
            from twisted.internet import reactor
        self._reactor = reactor
        self.avatar = avatar
        self.environ = {'PATH': '/bin:/usr/bin:/usr/local/bin'}
        self.pty = None
        self.ptyTuple = 0

    def addUTMPEntry(self, loggedIn=1):
        if not utmp:
            return
        ipAddress = self.avatar.conn.transport.transport.getPeer().host
        packedIp, = struct.unpack('L', socket.inet_aton(ipAddress))
        ttyName = self.ptyTuple[2][5:]
        t = time.time()
        t1 = int(t)
        t2 = int((t - t1) * 1000000.0)
        entry = utmp.UtmpEntry()
        entry.ut_type = loggedIn and utmp.USER_PROCESS or utmp.DEAD_PROCESS
        entry.ut_pid = self.pty.pid
        entry.ut_line = ttyName
        entry.ut_id = ttyName[-4:]
        entry.ut_tv = (t1, t2)
        if loggedIn:
            entry.ut_user = self.avatar.username
            entry.ut_host = socket.gethostbyaddr(ipAddress)[0]
            entry.ut_addr_v6 = (packedIp, 0, 0, 0)
        a = utmp.UtmpRecord(utmp.UTMP_FILE)
        a.pututline(entry)
        a.endutent()
        b = utmp.UtmpRecord(utmp.WTMP_FILE)
        b.pututline(entry)
        b.endutent()

    def getPty(self, term, windowSize, modes):
        self.environ['TERM'] = term
        self.winSize = windowSize
        self.modes = modes
        master, slave = pty.openpty()
        ttyname = os.ttyname(slave)
        self.environ['SSH_TTY'] = ttyname
        self.ptyTuple = (master, slave, ttyname)

    def openShell(self, proto):
        if not self.ptyTuple:
            self._log.error('tried to get shell without pty, failing')
            raise ConchError('no pty')
        uid, gid = self.avatar.getUserGroupId()
        homeDir = self.avatar.getHomeDir()
        shell = self.avatar.getShell()
        self.environ['USER'] = self.avatar.username
        self.environ['HOME'] = homeDir
        self.environ['SHELL'] = shell
        shellExec = os.path.basename(shell)
        peer = self.avatar.conn.transport.transport.getPeer()
        host = self.avatar.conn.transport.transport.getHost()
        self.environ['SSH_CLIENT'] = f'{peer.host} {peer.port} {host.port}'
        self.getPtyOwnership()
        self.pty = self._reactor.spawnProcess(proto, shell, [f'-{shellExec}'], self.environ, homeDir, uid, gid, usePTY=self.ptyTuple)
        self.addUTMPEntry()
        fcntl.ioctl(self.pty.fileno(), tty.TIOCSWINSZ, struct.pack('4H', *self.winSize))
        if self.modes:
            self.setModes()
        self.oldWrite = proto.transport.write
        proto.transport.write = self._writeHack
        self.avatar.conn.transport.transport.setTcpNoDelay(1)

    def execCommand(self, proto, cmd):
        uid, gid = self.avatar.getUserGroupId()
        homeDir = self.avatar.getHomeDir()
        shell = self.avatar.getShell() or '/bin/sh'
        self.environ['HOME'] = homeDir
        command = (shell, '-c', cmd)
        peer = self.avatar.conn.transport.transport.getPeer()
        host = self.avatar.conn.transport.transport.getHost()
        self.environ['SSH_CLIENT'] = f'{peer.host} {peer.port} {host.port}'
        if self.ptyTuple:
            self.getPtyOwnership()
        self.pty = self._reactor.spawnProcess(proto, shell, command, self.environ, homeDir, uid, gid, usePTY=self.ptyTuple or 0)
        if self.ptyTuple:
            self.addUTMPEntry()
            if self.modes:
                self.setModes()
        self.avatar.conn.transport.transport.setTcpNoDelay(1)

    def getPtyOwnership(self):
        ttyGid = os.stat(self.ptyTuple[2])[5]
        uid, gid = self.avatar.getUserGroupId()
        euid, egid = (os.geteuid(), os.getegid())
        os.setegid(0)
        os.seteuid(0)
        try:
            os.chown(self.ptyTuple[2], uid, ttyGid)
        finally:
            os.setegid(egid)
            os.seteuid(euid)

    def setModes(self):
        pty = self.pty
        attr = tty.tcgetattr(pty.fileno())
        for mode, modeValue in self.modes:
            if mode not in ttymodes.TTYMODES:
                continue
            ttyMode = ttymodes.TTYMODES[mode]
            if len(ttyMode) == 2:
                flag, ttyAttr = ttyMode
                if not hasattr(tty, ttyAttr):
                    continue
                ttyval = getattr(tty, ttyAttr)
                if modeValue:
                    attr[flag] = attr[flag] | ttyval
                else:
                    attr[flag] = attr[flag] & ~ttyval
            elif ttyMode == 'OSPEED':
                attr[tty.OSPEED] = getattr(tty, f'B{modeValue}')
            elif ttyMode == 'ISPEED':
                attr[tty.ISPEED] = getattr(tty, f'B{modeValue}')
            else:
                if not hasattr(tty, ttyMode):
                    continue
                ttyval = getattr(tty, ttyMode)
                attr[tty.CC][ttyval] = bytes((modeValue,))
        tty.tcsetattr(pty.fileno(), tty.TCSANOW, attr)

    def eofReceived(self):
        if self.pty:
            self.pty.closeStdin()

    def closed(self):
        if self.ptyTuple and os.path.exists(self.ptyTuple[2]):
            ttyGID = os.stat(self.ptyTuple[2])[5]
            os.chown(self.ptyTuple[2], 0, ttyGID)
        if self.pty:
            try:
                self.pty.signalProcess('HUP')
            except (OSError, ProcessExitedAlready):
                pass
            self.pty.loseConnection()
            self.addUTMPEntry(0)
        self._log.info('shell closed')

    def windowChanged(self, winSize):
        self.winSize = winSize
        fcntl.ioctl(self.pty.fileno(), tty.TIOCSWINSZ, struct.pack('4H', *self.winSize))

    def _writeHack(self, data):
        """
        Hack to send ignore messages when we aren't echoing.
        """
        if self.pty is not None:
            attr = tty.tcgetattr(self.pty.fileno())[3]
            if not attr & tty.ECHO and attr & tty.ICANON:
                self.avatar.conn.transport.sendIgnore('\x00' * (8 + len(data)))
        self.oldWrite(data)