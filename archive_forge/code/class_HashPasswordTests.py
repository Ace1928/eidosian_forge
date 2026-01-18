from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, task
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import IProtocolFactory
from twisted.python import failure
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words.protocols.jabber import error, ijabber, jid, xmlstream
from twisted.words.test.test_xmlstream import GenericXmlStreamFactoryTestsMixin
from twisted.words.xish import domish
class HashPasswordTests(unittest.TestCase):
    """
    Tests for L{xmlstream.hashPassword}.
    """

    def test_basic(self):
        """
        The sid and secret are concatenated to calculate sha1 hex digest.
        """
        hash = xmlstream.hashPassword('12345', 'secret')
        self.assertEqual('99567ee91b2c7cabf607f10cb9f4a3634fa820e0', hash)

    def test_sidNotUnicode(self):
        """
        The session identifier must be a unicode object.
        """
        self.assertRaises(TypeError, xmlstream.hashPassword, b'\xc2\xb92345', 'secret')

    def test_passwordNotUnicode(self):
        """
        The password must be a unicode object.
        """
        self.assertRaises(TypeError, xmlstream.hashPassword, '12345', b'secr\xc3\xa9t')

    def test_unicodeSecret(self):
        """
        The concatenated sid and password must be encoded to UTF-8 before hashing.
        """
        hash = xmlstream.hashPassword('12345', 'secrét')
        self.assertEqual('659bf88d8f8e179081f7f3b4a8e7d224652d2853', hash)