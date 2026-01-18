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
def addHostKey(self, hostname, key):
    """
        Add a new L{HashedEntry} to the key database.

        Note that you still need to call L{KnownHostsFile.save} if you wish
        these changes to be persisted.

        @param hostname: A hostname or IP address literal to associate with the
            new entry.
        @type hostname: L{bytes}

        @param key: The public key to associate with the new entry.
        @type key: L{Key}

        @return: The L{HashedEntry} that was added.
        @rtype: L{HashedEntry}
        """
    salt = secureRandom(20)
    keyType = key.sshType()
    entry = HashedEntry(salt, _hmacedString(salt, hostname), keyType, key, None)
    self._added.append(entry)
    return entry