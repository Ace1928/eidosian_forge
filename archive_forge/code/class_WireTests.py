from twisted.internet import address, defer, protocol, reactor
from twisted.protocols import portforward, wire
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class WireTests(unittest.TestCase):
    """
    Test wire protocols.
    """

    def test_echo(self):
        """
        Test wire.Echo protocol: send some data and check it send it back.
        """
        t = proto_helpers.StringTransport()
        a = wire.Echo()
        a.makeConnection(t)
        a.dataReceived(b'hello')
        a.dataReceived(b'world')
        a.dataReceived(b'how')
        a.dataReceived(b'are')
        a.dataReceived(b'you')
        self.assertEqual(t.value(), b'helloworldhowareyou')

    def test_who(self):
        """
        Test wire.Who protocol.
        """
        t = proto_helpers.StringTransport()
        a = wire.Who()
        a.makeConnection(t)
        self.assertEqual(t.value(), b'root\r\n')

    def test_QOTD(self):
        """
        Test wire.QOTD protocol.
        """
        t = proto_helpers.StringTransport()
        a = wire.QOTD()
        a.makeConnection(t)
        self.assertEqual(t.value(), b'An apple a day keeps the doctor away.\r\n')

    def test_discard(self):
        """
        Test wire.Discard protocol.
        """
        t = proto_helpers.StringTransport()
        a = wire.Discard()
        a.makeConnection(t)
        a.dataReceived(b'hello')
        a.dataReceived(b'world')
        a.dataReceived(b'how')
        a.dataReceived(b'are')
        a.dataReceived(b'you')
        self.assertEqual(t.value(), b'')