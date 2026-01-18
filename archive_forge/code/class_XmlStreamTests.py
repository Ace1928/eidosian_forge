from twisted.internet import protocol
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.xish import domish, utility, xmlstream
class XmlStreamTests(unittest.TestCase):

    def setUp(self):
        self.connectionLostMsg = 'no reason'
        self.outlist = []
        self.xmlstream = xmlstream.XmlStream()
        self.xmlstream.transport = self
        self.xmlstream.transport.write = self.outlist.append

    def loseConnection(self):
        """
        Stub loseConnection because we are a transport.
        """
        self.xmlstream.connectionLost(failure.Failure(Exception(self.connectionLostMsg)))

    def test_send(self):
        """
        Calling L{xmlstream.XmlStream.send} results in the data being written
        to the transport.
        """
        self.xmlstream.connectionMade()
        self.xmlstream.send(b'<root>')
        self.assertEqual(self.outlist[0], b'<root>')

    def test_receiveRoot(self):
        """
        Receiving the starttag of the root element results in stream start.
        """
        streamStarted = []

        def streamStartEvent(rootelem):
            streamStarted.append(None)
        self.xmlstream.addObserver(xmlstream.STREAM_START_EVENT, streamStartEvent)
        self.xmlstream.connectionMade()
        self.xmlstream.dataReceived('<root>')
        self.assertEqual(1, len(streamStarted))

    def test_receiveBadXML(self):
        """
        Receiving malformed XML results in an L{STREAM_ERROR_EVENT}.
        """
        streamError = []
        streamEnd = []

        def streamErrorEvent(reason):
            streamError.append(reason)

        def streamEndEvent(_):
            streamEnd.append(None)
        self.xmlstream.addObserver(xmlstream.STREAM_ERROR_EVENT, streamErrorEvent)
        self.xmlstream.addObserver(xmlstream.STREAM_END_EVENT, streamEndEvent)
        self.xmlstream.connectionMade()
        self.xmlstream.dataReceived('<root>')
        self.assertEqual(0, len(streamError))
        self.assertEqual(0, len(streamEnd))
        self.xmlstream.dataReceived('<child><unclosed></child>')
        self.assertEqual(1, len(streamError))
        self.assertTrue(streamError[0].check(domish.ParserError))
        self.assertEqual(1, len(streamEnd))

    def test_streamEnd(self):
        """
        Ending the stream fires a L{STREAM_END_EVENT}.
        """
        streamEnd = []

        def streamEndEvent(reason):
            streamEnd.append(reason)
        self.xmlstream.addObserver(xmlstream.STREAM_END_EVENT, streamEndEvent)
        self.xmlstream.connectionMade()
        self.loseConnection()
        self.assertEqual(1, len(streamEnd))
        self.assertIsInstance(streamEnd[0], failure.Failure)
        self.assertEqual(streamEnd[0].getErrorMessage(), self.connectionLostMsg)