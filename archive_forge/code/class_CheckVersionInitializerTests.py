from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
class CheckVersionInitializerTests(unittest.TestCase):

    def setUp(self):
        a = xmlstream.Authenticator()
        xs = xmlstream.XmlStream(a)
        self.init = client.CheckVersionInitializer(xs)

    def testSupported(self):
        """
        Test supported version number 1.0
        """
        self.init.xmlstream.version = (1, 0)
        self.init.initialize()

    def testNotSupported(self):
        """
        Test unsupported version number 0.0, and check exception.
        """
        self.init.xmlstream.version = (0, 0)
        exc = self.assertRaises(error.StreamError, self.init.initialize)
        self.assertEqual('unsupported-version', exc.condition)