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
@implementer(portal.IRealm)
class BaseFTPRealm:
    """
    Base class for simple FTP realms which provides an easy hook for specifying
    the home directory for each user.
    """

    def __init__(self, anonymousRoot):
        self.anonymousRoot = filepath.FilePath(anonymousRoot)

    def getHomeDirectory(self, avatarId):
        """
        Return a L{FilePath} representing the home directory of the given
        avatar.  Override this in a subclass.

        @param avatarId: A user identifier returned from a credentials checker.
        @type avatarId: C{str}

        @rtype: L{FilePath}
        """
        raise NotImplementedError(f'{self.__class__!r} did not override getHomeDirectory')

    def requestAvatar(self, avatarId, mind, *interfaces):
        for iface in interfaces:
            if iface is IFTPShell:
                if avatarId is checkers.ANONYMOUS:
                    avatar = FTPAnonymousShell(self.anonymousRoot)
                else:
                    avatar = FTPShell(self.getHomeDirectory(avatarId))
                return (IFTPShell, avatar, getattr(avatar, 'logout', lambda: None))
        raise NotImplementedError('Only IFTPShell interface is supported by this realm')