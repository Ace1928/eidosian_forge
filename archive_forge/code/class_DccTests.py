import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class DccTests(IRCTestCase):
    """
    Tests for C{dcc_*} methods.
    """

    def setUp(self):
        methods = ['dccDoSend', 'dccDoAcceptResume', 'dccDoResume', 'dccDoChat']
        self.user = 'Wolf!~wolf@yok.utu.fi'
        self.channel = '#twisted'
        self.client = CollectorClient(methods)

    def test_dccSend(self):
        """
        L{irc.IRCClient.dcc_SEND} invokes L{irc.IRCClient.dccDoSend}.
        """
        self.client.dcc_SEND(self.user, self.channel, 'foo.txt 127.0.0.1 1025')
        self.assertEqual(self.client.methods, [('dccDoSend', (self.user, '127.0.0.1', 1025, 'foo.txt', -1, ['foo.txt', '127.0.0.1', '1025']))])

    def test_dccSendNotImplemented(self):
        """
        L{irc.IRCClient.dccDoSend} is raises C{NotImplementedError}
        """
        client = irc.IRCClient()
        self.assertRaises(NotImplementedError, client.dccSend, 'username', None)

    def test_dccSendMalformedRequest(self):
        """
        L{irc.IRCClient.dcc_SEND} raises L{irc.IRCBadMessage} when it is passed
        a malformed query string.
        """
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_SEND, self.user, self.channel, 'foo')
        self.assertEqual(str(result), "malformed DCC SEND request: ['foo']")

    def test_dccSendIndecipherableAddress(self):
        """
        L{irc.IRCClient.dcc_SEND} raises L{irc.IRCBadMessage} when it is passed
        a query string that doesn't contain a valid address.
        """
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_SEND, self.user, self.channel, 'foo.txt #23 sd@d')
        self.assertEqual(str(result), "Indecipherable address '#23'")

    def test_dccSendIndecipherablePort(self):
        """
        L{irc.IRCClient.dcc_SEND} raises L{irc.IRCBadMessage} when it is passed
        a query string that doesn't contain a valid port number.
        """
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_SEND, self.user, self.channel, 'foo.txt 127.0.0.1 sd@d')
        self.assertEqual(str(result), "Indecipherable port 'sd@d'")

    def test_dccAccept(self):
        """
        L{irc.IRCClient.dcc_ACCEPT} invokes L{irc.IRCClient.dccDoAcceptResume}.
        """
        self.client.dcc_ACCEPT(self.user, self.channel, 'foo.txt 1025 2')
        self.assertEqual(self.client.methods, [('dccDoAcceptResume', (self.user, 'foo.txt', 1025, 2))])

    def test_dccAcceptMalformedRequest(self):
        """
        L{irc.IRCClient.dcc_ACCEPT} raises L{irc.IRCBadMessage} when it is
        passed a malformed query string.
        """
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_ACCEPT, self.user, self.channel, 'foo')
        self.assertEqual(str(result), "malformed DCC SEND ACCEPT request: ['foo']")

    def test_dccResume(self):
        """
        L{irc.IRCClient.dcc_RESUME} invokes L{irc.IRCClient.dccDoResume}.
        """
        self.client.dcc_RESUME(self.user, self.channel, 'foo.txt 1025 2')
        self.assertEqual(self.client.methods, [('dccDoResume', (self.user, 'foo.txt', 1025, 2))])

    def test_dccResumeMalformedRequest(self):
        """
        L{irc.IRCClient.dcc_RESUME} raises L{irc.IRCBadMessage} when it is
        passed a malformed query string.
        """
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_RESUME, self.user, self.channel, 'foo')
        self.assertEqual(str(result), "malformed DCC SEND RESUME request: ['foo']")

    def test_dccChat(self):
        """
        L{irc.IRCClient.dcc_CHAT} invokes L{irc.IRCClient.dccDoChat}.
        """
        self.client.dcc_CHAT(self.user, self.channel, 'foo.txt 127.0.0.1 1025')
        self.assertEqual(self.client.methods, [('dccDoChat', (self.user, self.channel, '127.0.0.1', 1025, ['foo.txt', '127.0.0.1', '1025']))])

    def test_dccChatMalformedRequest(self):
        """
        L{irc.IRCClient.dcc_CHAT} raises L{irc.IRCBadMessage} when it is
        passed a malformed query string.
        """
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_CHAT, self.user, self.channel, 'foo')
        self.assertEqual(str(result), "malformed DCC CHAT request: ['foo']")

    def test_dccChatIndecipherablePort(self):
        """
        L{irc.IRCClient.dcc_CHAT} raises L{irc.IRCBadMessage} when it is passed
        a query string that doesn't contain a valid port number.
        """
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_CHAT, self.user, self.channel, 'foo.txt 127.0.0.1 sd@d')
        self.assertEqual(str(result), "Indecipherable port 'sd@d'")