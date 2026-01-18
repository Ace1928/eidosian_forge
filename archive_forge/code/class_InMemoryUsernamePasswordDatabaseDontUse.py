import os
from typing import Any, Dict, Optional, Tuple, Union
from zope.interface import Attribute, Interface, implementer
from twisted.cred import error
from twisted.cred.credentials import (
from twisted.internet import defer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import failure
@implementer(ICredentialsChecker)
class InMemoryUsernamePasswordDatabaseDontUse:
    """
    An extremely simple credentials checker.

    This is only of use in one-off test programs or examples which don't
    want to focus too much on how credentials are verified.

    You really don't want to use this for anything else.  It is, at best, a
    toy.  If you need a simple credentials checker for a real application,
    see L{FilePasswordDB}.

    @cvar credentialInterfaces: Tuple of L{IUsernamePassword} and
    L{IUsernameHashedPassword}.

    @ivar users: Mapping of usernames to passwords.
    @type users: L{dict} mapping L{bytes} to L{bytes}
    """
    credentialInterfaces = (IUsernamePassword, IUsernameHashedPassword)

    def __init__(self, **users: bytes) -> None:
        """
        Initialize the in-memory database.

        For example::

            db = InMemoryUsernamePasswordDatabaseDontUse(
                user1=b'sesame',
                user2=b'hunter2',
            )

        @param users: Usernames and passwords to seed the database with.
        Each username given as a keyword is encoded to L{bytes} as ASCII.
        Passwords must be given as L{bytes}.
        @type users: L{dict} of L{str} to L{bytes}
        """
        self.users = {x.encode('ascii'): y for x, y in users.items()}

    def addUser(self, username: bytes, password: bytes) -> None:
        """
        Set a user's password.

        @param username: Name of the user.
        @type username: L{bytes}

        @param password: Password to associate with the username.
        @type password: L{bytes}
        """
        self.users[username] = password

    def _cbPasswordMatch(self, matched, username):
        if matched:
            return username
        else:
            return failure.Failure(error.UnauthorizedLogin())

    def requestAvatarId(self, credentials):
        if credentials.username in self.users:
            return defer.maybeDeferred(credentials.checkPassword, self.users[credentials.username]).addCallback(self._cbPasswordMatch, credentials.username)
        else:
            return defer.fail(error.UnauthorizedLogin())