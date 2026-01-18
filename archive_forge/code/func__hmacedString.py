import hmac
import sys
from binascii import Error as DecodeError, a2b_base64, b2a_base64
from contextlib import closing
from hashlib import sha1
from zope.interface import implementer
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.conch.ssh.keys import BadKeyError, FingerprintFormats, Key
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString
from twisted.python.randbytes import secureRandom
from twisted.python.util import FancyEqMixin
def _hmacedString(key, string):
    """
    Return the SHA-1 HMAC hash of the given key and string.

    @param key: The HMAC key.
    @type key: L{bytes}

    @param string: The string to be hashed.
    @type string: L{bytes}

    @return: The keyed hash value.
    @rtype: L{bytes}
    """
    hash = hmac.HMAC(key, digestmod=sha1)
    if isinstance(string, str):
        string = string.encode('utf-8')
    hash.update(string)
    return hash.digest()