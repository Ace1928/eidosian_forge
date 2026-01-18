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
def _lookupUser(userdb: UserDB, username: bytes) -> UserRecord:
    """
    Lookup a user by name in a L{pwd}-style database.

    @param userdb: The user database.

    @param username: Identifying name in bytes. This will be decoded according
    to the filesystem encoding, as the L{pwd} module does internally.

    @raises KeyError: when the user doesn't exist
    """
    return userdb.getpwnam(username.decode(sys.getfilesystemencoding()))