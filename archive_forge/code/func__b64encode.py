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
def _b64encode(s):
    """
    Encode a binary string as base64 with no trailing newline.

    @param s: The string to encode.
    @type s: L{bytes}

    @return: The base64-encoded string.
    @rtype: L{bytes}
    """
    return b2a_base64(s).strip()