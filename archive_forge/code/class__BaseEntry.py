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
class _BaseEntry:
    """
    Abstract base of both hashed and non-hashed entry objects, since they
    represent keys and key types the same way.

    @ivar keyType: The type of the key; either ssh-dss or ssh-rsa.
    @type keyType: L{bytes}

    @ivar publicKey: The server public key indicated by this line.
    @type publicKey: L{twisted.conch.ssh.keys.Key}

    @ivar comment: Trailing garbage after the key line.
    @type comment: L{bytes}
    """

    def __init__(self, keyType, publicKey, comment):
        self.keyType = keyType
        self.publicKey = publicKey
        self.comment = comment

    def matchesKey(self, keyObject):
        """
        Check to see if this entry matches a given key object.

        @param keyObject: A public key object to check.
        @type keyObject: L{Key}

        @return: C{True} if this entry's key matches C{keyObject}, C{False}
            otherwise.
        @rtype: L{bool}
        """
        return self.publicKey == keyObject