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
class CmdNotImplementedForArgError(FTPCmdError):
    """
    Raised when the handling of a parameter for a command is not implemented by
    the server.
    """
    errorCode = CMD_NOT_IMPLMNTD_FOR_PARAM