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
class SystemFTPRealm(BaseFTPRealm):
    """
    L{SystemFTPRealm} uses system user account information to decide what the
    home directory for a particular avatarId is.

    This works on POSIX but probably is not reliable on Windows.
    """

    def getHomeDirectory(self, avatarId):
        """
        Return the system-defined home directory of the system user account
        with the name C{avatarId}.
        """
        path = os.path.expanduser('~' + avatarId)
        if path.startswith('~'):
            raise cred_error.UnauthorizedLogin()
        return filepath.FilePath(path)