from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
class MakeMessageTests(unittest.TestCase):

    def testRequest(self):
        r = sip.Request('INVITE', 'sip:foo')
        r.addHeader('foo', 'bar')
        self.assertEqual(r.toString(), 'INVITE sip:foo SIP/2.0\r\nFoo: bar\r\n\r\n')

    def testResponse(self):
        r = sip.Response(200, 'OK')
        r.addHeader('foo', 'bar')
        r.addHeader('Content-Length', '4')
        r.bodyDataReceived('1234')
        self.assertEqual(r.toString(), 'SIP/2.0 200 OK\r\nFoo: bar\r\nContent-Length: 4\r\n\r\n1234')

    def testStatusCode(self):
        r = sip.Response(200)
        self.assertEqual(r.toString(), 'SIP/2.0 200 OK\r\n\r\n')