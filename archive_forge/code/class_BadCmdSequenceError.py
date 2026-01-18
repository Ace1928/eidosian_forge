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
class BadCmdSequenceError(FTPCmdError):
    """
    Raised when a client sends a series of commands in an illogical sequence.
    """
    errorCode = BAD_CMD_SEQ