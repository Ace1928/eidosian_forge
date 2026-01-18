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
@implementer(IAuthorizedKeysDB)
class UNIXAuthorizedKeysFiles:
    """
    Object that provides SSH public keys based on public keys listed in
    authorized_keys and authorized_keys2 files in UNIX user .ssh/ directories.
    If any of the files cannot be read, a message is logged but that file is
    otherwise ignored.

    @since: 15.0
    """
    _userdb: UserDB

    def __init__(self, userdb: Optional[UserDB]=None, parseKey: Callable[[bytes], keys.Key]=keys.Key.fromString):
        """
        Initializes a new L{UNIXAuthorizedKeysFiles}.

        @param userdb: access to the Unix user account and password database
            (default is the Python module L{pwd}, if available)

        @param parseKey: a callable that takes a string and returns a
            L{twisted.conch.ssh.keys.Key}, mainly to be used for testing.  The
            default is L{twisted.conch.ssh.keys.Key.fromString}.
        """
        if userdb is not None:
            self._userdb = userdb
        elif pwd is not None:
            self._userdb = pwd
        else:
            raise ValueError('No pwd module found, and no userdb argument passed.')
        self._parseKey = parseKey

    def getAuthorizedKeys(self, username: bytes) -> Iterable[keys.Key]:
        try:
            passwd = _lookupUser(self._userdb, username)
        except KeyError:
            return ()
        root = FilePath(passwd.pw_dir).child('.ssh')
        files = ['authorized_keys', 'authorized_keys2']
        return _keysFromFilepaths((root.child(f) for f in files), self._parseKey)