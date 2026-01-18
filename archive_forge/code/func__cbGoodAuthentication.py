import binascii
import errno
import sys
from base64 import decodebytes
from typing import IO, Any, Callable, Iterable, Iterator, Mapping, Optional, Tuple, cast
from zope.interface import Interface, implementer, providedBy
from incremental import Version
from typing_extensions import Literal, Protocol
from twisted.conch import error
from twisted.conch.ssh import keys
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import ISSHPrivateKey, IUsernamePassword
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer
from twisted.logger import Logger
from twisted.plugins.cred_unix import verifyCryptedPassword
from twisted.python import failure, reflect
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.filepath import FilePath
from twisted.python.util import runAsEffectiveUser
def _cbGoodAuthentication(self, avatarId, credentials):
    """
        Called if a checker has verified the credentials.  We call our
        L{areDone} method to see if the whole of the successful authentications
        are enough.  If they are, we return the avatar ID returned by the first
        checker.
        """
    if avatarId not in self.successfulCredentials:
        self.successfulCredentials[avatarId] = []
    self.successfulCredentials[avatarId].append(credentials)
    if self.areDone(avatarId):
        del self.successfulCredentials[avatarId]
        return avatarId
    else:
        raise error.NotEnoughAuthentication()