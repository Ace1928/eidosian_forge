import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
class HashedEntryTests(EntryTestsMixin, ComparisonTestsMixin, TestCase):
    """
    Tests for L{HashedEntry}.

    This suite doesn't include any tests for host/IP pairs because hashed
    entries store IP addresses the same way as hostnames and does not support
    comma-separated lists.  (If you hash the IP and host together you can't
    tell if you've got the key already for one or the other.)
    """
    hashedLine = sampleHashedLine

    def setUp(self):
        """
        Set 'entry' to a sample hashed entry for twistedmatrix.com with
        sampleKey as its key.
        """
        self.entry = HashedEntry.fromString(self.hashedLine)

    def test_toString(self):
        """
        L{HashedEntry.toString} generates the serialized OpenSSL format string
        for the entry, sans the newline.
        """
        self.assertEqual(self.entry.toString(), self.hashedLine.rstrip(b'\n'))

    def test_equality(self):
        """
        Two L{HashedEntry} instances compare equal if and only if they represent
        the same host and key in exactly the same way: the host salt, host hash,
        public key type, public key, and comment fields must all be equal.
        """
        hostSalt = b'gJbSEPBG9ZSBoZpHNtZBD1bHKBA'
        hostHash = b'bQv+0Xa0dByrwkA1EB0E7Xop/Fo'
        publicKey = Key.fromString(sampleKey)
        keyType = networkString(publicKey.type())
        comment = b'hello, world'
        entry = HashedEntry(hostSalt, hostHash, keyType, publicKey, comment)
        duplicate = HashedEntry(hostSalt, hostHash, keyType, publicKey, comment)
        self.assertNormalEqualityImplementation(entry, duplicate, HashedEntry(hostSalt[::-1], hostHash, keyType, publicKey, comment))
        self.assertNormalEqualityImplementation(entry, duplicate, HashedEntry(hostSalt, hostHash[::-1], keyType, publicKey, comment))
        self.assertNormalEqualityImplementation(entry, duplicate, HashedEntry(hostSalt, hostHash, keyType[::-1], publicKey, comment))
        self.assertNormalEqualityImplementation(entry, duplicate, HashedEntry(hostSalt, hostHash, keyType, Key.fromString(otherSampleKey), comment))
        self.assertNormalEqualityImplementation(entry, duplicate, HashedEntry(hostSalt, hostHash, keyType, publicKey, comment[::-1]))