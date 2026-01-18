import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
class FTPOverflowProtocol(basic.LineReceiver):
    """FTP mini-protocol for when there are too many connections."""
    _encoding = 'latin-1'

    def connectionMade(self):
        self.sendLine(RESPONSE[TOO_MANY_CONNECTIONS].encode(self._encoding))
        self.transport.loseConnection()