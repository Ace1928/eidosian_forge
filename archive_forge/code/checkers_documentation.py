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

        Checks whether the credentials themselves are valid, now that we know
        if the key matches the user.

        @param pubKey: the key in the credentials (just to prevent it from
            having to be calculated again)
        @type pubKey: L{twisted.conch.ssh.keys.Key}

        @param credentials: the credentials offered by the user
        @type credentials: L{ISSHPrivateKey} provider

        @raise UnauthorizedLogin: If the key signature is invalid or there
            was any error verifying the signature.

        @return: The user's username, if authentication was successful
        @rtype: L{bytes}
        