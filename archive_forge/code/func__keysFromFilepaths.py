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
def _keysFromFilepaths(filepaths: Iterable[FilePath[Any]], parseKey: Callable[[bytes], keys.Key]) -> Iterable[keys.Key]:
    """
    Helper function that turns an iterable of filepaths into a generator of
    keys.  If any file cannot be read, a message is logged but it is
    otherwise ignored.

    @param filepaths: iterable of L{twisted.python.filepath.FilePath}.
    @type filepaths: iterable

    @param parseKey: a callable that takes a string and returns a
        L{twisted.conch.ssh.keys.Key}
    @type parseKey: L{callable}

    @return: generator of L{twisted.conch.ssh.keys.Key}

    @since: 15.0
    """
    for fp in filepaths:
        if fp.exists():
            try:
                with fp.open() as f:
                    yield from readAuthorizedKeyFile(f, parseKey)
            except OSError as e:
                _log.error('Unable to read {path!r}: {error!s}', path=fp.path, error=e)