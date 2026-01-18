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
class FTPRealm(BaseFTPRealm):
    """
    @type anonymousRoot: L{twisted.python.filepath.FilePath}
    @ivar anonymousRoot: Root of the filesystem to which anonymous
        users will be granted access.

    @type userHome: L{filepath.FilePath}
    @ivar userHome: Root of the filesystem containing user home directories.
    """

    def __init__(self, anonymousRoot, userHome='/home'):
        BaseFTPRealm.__init__(self, anonymousRoot)
        self.userHome = filepath.FilePath(userHome)

    def getHomeDirectory(self, avatarId):
        """
        Use C{avatarId} as a single path segment to construct a child of
        C{self.userHome} and return that child.
        """
        return self.userHome.child(avatarId)