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
@implementer(ISFTPServer)
class SFTPServerForUnixConchUser:

    def __init__(self, avatar):
        self.avatar = avatar

    def _setAttrs(self, path, attrs):
        """
        NOTE: this function assumes it runs as the logged-in user:
        i.e. under _runAsUser()
        """
        if 'uid' in attrs and 'gid' in attrs:
            os.chown(path, attrs['uid'], attrs['gid'])
        if 'permissions' in attrs:
            os.chmod(path, attrs['permissions'])
        if 'atime' in attrs and 'mtime' in attrs:
            os.utime(path, (attrs['atime'], attrs['mtime']))

    def _getAttrs(self, s):
        return {'size': s.st_size, 'uid': s.st_uid, 'gid': s.st_gid, 'permissions': s.st_mode, 'atime': int(s.st_atime), 'mtime': int(s.st_mtime)}

    def _absPath(self, path):
        home = self.avatar.getHomeDir()
        return os.path.join(nativeString(home.path), nativeString(path))

    def gotVersion(self, otherVersion, extData):
        return {}

    def openFile(self, filename, flags, attrs):
        return UnixSFTPFile(self, self._absPath(filename), flags, attrs)

    def removeFile(self, filename):
        filename = self._absPath(filename)
        return self.avatar._runAsUser(os.remove, filename)

    def renameFile(self, oldpath, newpath):
        oldpath = self._absPath(oldpath)
        newpath = self._absPath(newpath)
        return self.avatar._runAsUser(os.rename, oldpath, newpath)

    def makeDirectory(self, path, attrs):
        path = self._absPath(path)
        return self.avatar._runAsUser([(os.mkdir, (path,)), (self._setAttrs, (path, attrs))])

    def removeDirectory(self, path):
        path = self._absPath(path)
        self.avatar._runAsUser(os.rmdir, path)

    def openDirectory(self, path):
        return UnixSFTPDirectory(self, self._absPath(path))

    def getAttrs(self, path, followLinks):
        path = self._absPath(path)
        if followLinks:
            s = self.avatar._runAsUser(os.stat, path)
        else:
            s = self.avatar._runAsUser(os.lstat, path)
        return self._getAttrs(s)

    def setAttrs(self, path, attrs):
        path = self._absPath(path)
        self.avatar._runAsUser(self._setAttrs, path, attrs)

    def readLink(self, path):
        path = self._absPath(path)
        return self.avatar._runAsUser(os.readlink, path)

    def makeLink(self, linkPath, targetPath):
        linkPath = self._absPath(linkPath)
        targetPath = self._absPath(targetPath)
        return self.avatar._runAsUser(os.symlink, targetPath, linkPath)

    def realPath(self, path):
        return os.path.realpath(self._absPath(path))

    def extendedRequest(self, extName, extData):
        raise NotImplementedError