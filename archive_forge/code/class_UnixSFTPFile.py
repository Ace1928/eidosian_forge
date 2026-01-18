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
@implementer(ISFTPFile)
class UnixSFTPFile:

    def __init__(self, server, filename, flags, attrs):
        self.server = server
        openFlags = 0
        if flags & FXF_READ == FXF_READ and flags & FXF_WRITE == 0:
            openFlags = os.O_RDONLY
        if flags & FXF_WRITE == FXF_WRITE and flags & FXF_READ == 0:
            openFlags = os.O_WRONLY
        if flags & FXF_WRITE == FXF_WRITE and flags & FXF_READ == FXF_READ:
            openFlags = os.O_RDWR
        if flags & FXF_APPEND == FXF_APPEND:
            openFlags |= os.O_APPEND
        if flags & FXF_CREAT == FXF_CREAT:
            openFlags |= os.O_CREAT
        if flags & FXF_TRUNC == FXF_TRUNC:
            openFlags |= os.O_TRUNC
        if flags & FXF_EXCL == FXF_EXCL:
            openFlags |= os.O_EXCL
        if 'permissions' in attrs:
            mode = attrs['permissions']
            del attrs['permissions']
        else:
            mode = 511
        fd = server.avatar._runAsUser(os.open, filename, openFlags, mode)
        if attrs:
            server.avatar._runAsUser(server._setAttrs, filename, attrs)
        self.fd = fd

    def close(self):
        return self.server.avatar._runAsUser(os.close, self.fd)

    def readChunk(self, offset, length):
        return self.server.avatar._runAsUser([(os.lseek, (self.fd, offset, 0)), (os.read, (self.fd, length))])

    def writeChunk(self, offset, data):
        return self.server.avatar._runAsUser([(os.lseek, (self.fd, offset, 0)), (os.write, (self.fd, data))])

    def getAttrs(self):
        s = self.server.avatar._runAsUser(os.fstat, self.fd)
        return self.server._getAttrs(s)

    def setAttrs(self, attrs):
        raise NotImplementedError