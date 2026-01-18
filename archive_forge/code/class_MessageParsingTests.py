from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
class MessageParsingTests(unittest.TestCase):

    def setUp(self):
        self.l = []
        self.parser = sip.MessagesParser(self.l.append)

    def feedMessage(self, message):
        self.parser.dataReceived(message)
        self.parser.dataDone()

    def validateMessage(self, m, method, uri, headers, body):
        """
        Validate Requests.
        """
        self.assertEqual(m.method, method)
        self.assertEqual(m.uri.toString(), uri)
        self.assertEqual(m.headers, headers)
        self.assertEqual(m.body, body)
        self.assertEqual(m.finished, 1)

    def testSimple(self):
        l = self.l
        self.feedMessage(request1)
        self.assertEqual(len(l), 1)
        self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['4']}, 'abcd')

    def testTwoMessages(self):
        l = self.l
        self.feedMessage(request1)
        self.feedMessage(request2)
        self.assertEqual(len(l), 2)
        self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['4']}, 'abcd')
        self.validateMessage(l[1], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe']}, '1234')

    def testGarbage(self):
        l = self.l
        self.feedMessage(request3)
        self.assertEqual(len(l), 1)
        self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['4']}, '1234')

    def testThreeInOne(self):
        l = self.l
        self.feedMessage(request4)
        self.assertEqual(len(l), 3)
        self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['0']}, '')
        self.validateMessage(l[1], 'INVITE', 'sip:loop', {'from': ['foo'], 'to': ['bar'], 'content-length': ['4']}, 'abcd')
        self.validateMessage(l[2], 'INVITE', 'sip:loop', {'from': ['foo'], 'to': ['bar'], 'content-length': ['4']}, '1234')

    def testShort(self):
        l = self.l
        self.feedMessage(request_short)
        self.assertEqual(len(l), 1)
        self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['4']}, 'abcd')

    def testSimpleResponse(self):
        l = self.l
        self.feedMessage(response1)
        self.assertEqual(len(l), 1)
        m = l[0]
        self.assertEqual(m.code, 200)
        self.assertEqual(m.phrase, 'OK')
        self.assertEqual(m.headers, {'from': ['foo'], 'to': ['bar'], 'content-length': ['0']})
        self.assertEqual(m.body, '')
        self.assertEqual(m.finished, 1)

    def test_multiLine(self):
        """
        A header may be split across multiple lines.  Subsequent lines begin
        with C{" "} or C{"\\t"}.
        """
        l = self.l
        self.feedMessage(response_multiline)
        self.assertEqual(len(l), 1)
        m = l[0]
        self.assertEqual(m.headers['via'][0], 'SIP/2.0/UDP server10.biloxi.com;branch=z9hG4bKnashds8;received=192.0.2.3')
        self.assertEqual(m.headers['via'][1], 'SIP/2.0/UDP bigbox3.site3.atlanta.com;branch=z9hG4bK77ef4c2312983.1;received=192.0.2.2')
        self.assertEqual(m.headers['via'][2], 'SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds ;received=192.0.2.1')