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
def _sanityCheckKey(self, credentials):
    """
        Checks whether the provided credentials are a valid SSH key with a
        signature (does not actually verify the signature).

        @param credentials: the credentials offered by the user
        @type credentials: L{ISSHPrivateKey} provider

        @raise ValidPublicKey: the credentials do not include a signature. See
            L{error.ValidPublicKey} for more information.

        @raise BadKeyError: The key included with the credentials is not
            recognized as a key.

        @return: the key in the credentials
        @rtype: L{twisted.conch.ssh.keys.Key}
        """
    if not credentials.signature:
        raise error.ValidPublicKey()
    return keys.Key.fromString(credentials.blob)