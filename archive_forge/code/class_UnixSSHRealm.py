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
@implementer(portal.IRealm)
class UnixSSHRealm:

    def requestAvatar(self, username: bytes | Tuple[()], mind: object, *interfaces: portal._InterfaceItself) -> Tuple[portal._InterfaceItself, UnixConchUser, Callable[[], None]]:
        if not isinstance(username, bytes):
            raise LoginDenied('UNIX SSH realm does not authorize anonymous sessions.')
        user = UnixConchUser(username.decode())
        return (interfaces[0], user, user.logout)