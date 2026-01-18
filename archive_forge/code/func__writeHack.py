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
def _writeHack(self, data):
    """
        Hack to send ignore messages when we aren't echoing.
        """
    if self.pty is not None:
        attr = tty.tcgetattr(self.pty.fileno())[3]
        if not attr & tty.ECHO and attr & tty.ICANON:
            self.avatar.conn.transport.sendIgnore('\x00' * (8 + len(data)))
    self.oldWrite(data)