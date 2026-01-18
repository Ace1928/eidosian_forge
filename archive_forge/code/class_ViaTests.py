from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
class ViaTests(unittest.TestCase):

    def checkRoundtrip(self, v):
        s = v.toString()
        self.assertEqual(s, sip.parseViaHeader(s).toString())

    def testExtraWhitespace(self):
        v1 = sip.parseViaHeader('SIP/2.0/UDP 192.168.1.1:5060')
        v2 = sip.parseViaHeader('SIP/2.0/UDP     192.168.1.1:5060')
        self.assertEqual(v1.transport, v2.transport)
        self.assertEqual(v1.host, v2.host)
        self.assertEqual(v1.port, v2.port)

    def test_complex(self):
        """
        Test parsing a Via header with one of everything.
        """
        s = 'SIP/2.0/UDP first.example.com:4000;ttl=16;maddr=224.2.0.1 ;branch=a7c6a8dlze (Example)'
        v = sip.parseViaHeader(s)
        self.assertEqual(v.transport, 'UDP')
        self.assertEqual(v.host, 'first.example.com')
        self.assertEqual(v.port, 4000)
        self.assertIsNone(v.rport)
        self.assertIsNone(v.rportValue)
        self.assertFalse(v.rportRequested)
        self.assertEqual(v.ttl, 16)
        self.assertEqual(v.maddr, '224.2.0.1')
        self.assertEqual(v.branch, 'a7c6a8dlze')
        self.assertEqual(v.hidden, 0)
        self.assertEqual(v.toString(), 'SIP/2.0/UDP first.example.com:4000;ttl=16;branch=a7c6a8dlze;maddr=224.2.0.1')
        self.checkRoundtrip(v)

    def test_simple(self):
        """
        Test parsing a simple Via header.
        """
        s = 'SIP/2.0/UDP example.com;hidden'
        v = sip.parseViaHeader(s)
        self.assertEqual(v.transport, 'UDP')
        self.assertEqual(v.host, 'example.com')
        self.assertEqual(v.port, 5060)
        self.assertIsNone(v.rport)
        self.assertIsNone(v.rportValue)
        self.assertFalse(v.rportRequested)
        self.assertIsNone(v.ttl)
        self.assertIsNone(v.maddr)
        self.assertIsNone(v.branch)
        self.assertTrue(v.hidden)
        self.assertEqual(v.toString(), 'SIP/2.0/UDP example.com:5060;hidden')
        self.checkRoundtrip(v)

    def testSimpler(self):
        v = sip.Via('example.com')
        self.checkRoundtrip(v)

    def test_deprecatedRPort(self):
        """
        Setting rport to True is deprecated, but still produces a Via header
        with the expected properties.
        """
        v = sip.Via('foo.bar', rport=True)
        warnings = self.flushWarnings(offendingFunctions=[self.test_deprecatedRPort])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['message'], 'rport=True is deprecated since Twisted 9.0.')
        self.assertEqual(warnings[0]['category'], DeprecationWarning)
        self.assertEqual(v.toString(), 'SIP/2.0/UDP foo.bar:5060;rport')
        self.assertTrue(v.rport)
        self.assertTrue(v.rportRequested)
        self.assertIsNone(v.rportValue)

    def test_rport(self):
        """
        An rport setting of None should insert the parameter with no value.
        """
        v = sip.Via('foo.bar', rport=None)
        self.assertEqual(v.toString(), 'SIP/2.0/UDP foo.bar:5060;rport')
        self.assertTrue(v.rportRequested)
        self.assertIsNone(v.rportValue)

    def test_rportValue(self):
        """
        An rport numeric setting should insert the parameter with the number
        value given.
        """
        v = sip.Via('foo.bar', rport=1)
        self.assertEqual(v.toString(), 'SIP/2.0/UDP foo.bar:5060;rport=1')
        self.assertFalse(v.rportRequested)
        self.assertEqual(v.rportValue, 1)
        self.assertEqual(v.rport, 1)

    def testNAT(self):
        s = 'SIP/2.0/UDP 10.0.0.1:5060;received=22.13.1.5;rport=12345'
        v = sip.parseViaHeader(s)
        self.assertEqual(v.transport, 'UDP')
        self.assertEqual(v.host, '10.0.0.1')
        self.assertEqual(v.port, 5060)
        self.assertEqual(v.received, '22.13.1.5')
        self.assertEqual(v.rport, 12345)
        self.assertNotEqual(v.toString().find('rport=12345'), -1)

    def test_unknownParams(self):
        """
        Parsing and serializing Via headers with unknown parameters should work.
        """
        s = 'SIP/2.0/UDP example.com:5060;branch=a12345b;bogus;pie=delicious'
        v = sip.parseViaHeader(s)
        self.assertEqual(v.toString(), s)