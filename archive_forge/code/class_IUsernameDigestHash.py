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
class IUsernameDigestHash(ICredentials):
    """
    This credential is used when a CredentialChecker has access to the hash
    of the username:realm:password as in an Apache .htdigest file.
    """

    def checkHash(digestHash):
        """
        @param digestHash: The hashed username:realm:password to check against.

        @return: C{True} if the credentials represented by this object match
            the given hash, C{False} if they do not, or a L{Deferred} which
            will be called back with one of these values.
        """