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
@implementer(ICredentialsChecker)
class UNIXPasswordDatabase:
    """
    A checker which validates users out of the UNIX password databases, or
    databases of a compatible format.

    @ivar _getByNameFunctions: a C{list} of functions which are called in order
        to validate a user.  The default value is such that the C{/etc/passwd}
        database will be tried first, followed by the C{/etc/shadow} database.
    """
    credentialInterfaces = (IUsernamePassword,)

    def __init__(self, getByNameFunctions=None):
        if getByNameFunctions is None:
            getByNameFunctions = [_pwdGetByName, _shadowGetByName]
        self._getByNameFunctions = getByNameFunctions

    def requestAvatarId(self, credentials):
        username = credentials.username.decode(sys.getfilesystemencoding())
        password = credentials.password.decode(sys.getfilesystemencoding())
        for func in self._getByNameFunctions:
            try:
                pwnam = func(username)
            except KeyError:
                return defer.fail(UnauthorizedLogin('invalid username'))
            else:
                if pwnam is not None:
                    crypted = pwnam[1]
                    if crypted == '':
                        continue
                    if verifyCryptedPassword(crypted, password):
                        return defer.succeed(credentials.username)
        return defer.fail(UnauthorizedLogin('unable to verify password'))