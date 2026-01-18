from typing import Dict, List, Tuple
from twisted.internet.testing import StringTransport
from twisted.protocols import postfix
from twisted.trial import unittest
class PostfixTCPMapServerTestCase:
    data: Dict[bytes, bytes] = {}
    chat: List[Tuple[bytes, bytes]] = []

    def test_chat(self):
        """
        Test that I{get} and I{put} commands are responded to correctly by
        L{postfix.PostfixTCPMapServer} when its factory is an instance of
        L{postifx.PostfixTCPMapDictServerFactory}.
        """
        factory = postfix.PostfixTCPMapDictServerFactory(self.data)
        transport = StringTransport()
        protocol = postfix.PostfixTCPMapServer()
        protocol.service = factory
        protocol.factory = factory
        protocol.makeConnection(transport)
        for input, expected_output in self.chat:
            protocol.lineReceived(input)
            self.assertEqual(transport.value(), expected_output, 'For %r, expected %r but got %r' % (input, expected_output, transport.value()))
            transport.clear()
        protocol.setTimeout(None)

    def test_deferredChat(self):
        """
        Test that I{get} and I{put} commands are responded to correctly by
        L{postfix.PostfixTCPMapServer} when its factory is an instance of
        L{postifx.PostfixTCPMapDeferringDictServerFactory}.
        """
        factory = postfix.PostfixTCPMapDeferringDictServerFactory(self.data)
        transport = StringTransport()
        protocol = postfix.PostfixTCPMapServer()
        protocol.service = factory
        protocol.factory = factory
        protocol.makeConnection(transport)
        for input, expected_output in self.chat:
            protocol.lineReceived(input)
            self.assertEqual(transport.value(), expected_output, 'For {!r}, expected {!r} but got {!r}'.format(input, expected_output, transport.value()))
            transport.clear()
        protocol.setTimeout(None)

    def test_getException(self):
        """
        If the factory throws an exception,
        error code 400 must be returned.
        """

        class ErrorFactory:
            """
            Factory that raises an error on key lookup.
            """

            def get(self, key):
                raise Exception('This is a test error')
        server = postfix.PostfixTCPMapServer()
        server.factory = ErrorFactory()
        server.transport = StringTransport()
        server.lineReceived(b'get example')
        self.assertEqual(server.transport.value(), b'400 This is a test error\n')