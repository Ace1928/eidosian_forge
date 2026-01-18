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
class FTPCmdError(Exception):
    """
    Generic exception for FTP commands.
    """

    def __init__(self, *msg):
        Exception.__init__(self, *msg)
        self.errorMessage = msg

    def response(self):
        """
        Generate a FTP response message for this error.
        """
        return RESPONSE[self.errorCode] % self.errorMessage