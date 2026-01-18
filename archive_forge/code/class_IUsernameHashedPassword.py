import base64
import hmac
import random
import re
import time
from binascii import hexlify
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred import error
from twisted.cred._digest import calcHA1, calcHA2, calcResponse
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.randbytes import secureRandom
from twisted.python.versions import Version
class IUsernameHashedPassword(ICredentials):
    """
    I encapsulate a username and a hashed password.

    This credential is used when a hashed password is received from the
    party requesting authentication.  CredentialCheckers which check this
    kind of credential must store the passwords in plaintext (or as
    password-equivalent hashes) form so that they can be hashed in a manner
    appropriate for the particular credentials class.

    @type username: L{bytes}
    @ivar username: The username associated with these credentials.
    """

    def checkPassword(password):
        """
        Validate these credentials against the correct password.

        @type password: L{bytes}
        @param password: The correct, plaintext password against which to
        check.

        @rtype: C{bool} or L{Deferred}
        @return: C{True} if the credentials represented by this object match the
            given password, C{False} if they do not, or a L{Deferred} which will
            be called back with one of these values.
        """