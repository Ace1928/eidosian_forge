import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class CTCPQueryTests(IRCTestCase):
    """
    Tests for the C{ctcpQuery_*} methods.
    """

    def setUp(self):
        self.user = 'Wolf!~wolf@yok.utu.fi'
        self.channel = '#twisted'
        self.client = CollectorClient(['ctcpMakeReply'])

    def test_ctcpQuery_PING(self):
        """
        L{IRCClient.ctcpQuery_PING} calls L{IRCClient.ctcpMakeReply} with the
        correct args.
        """
        self.client.ctcpQuery_PING(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('PING', 'data')]))])

    def test_ctcpQuery_FINGER(self):
        """
        L{IRCClient.ctcpQuery_FINGER} calls L{IRCClient.ctcpMakeReply} with the
        correct args.
        """
        self.client.fingerReply = 'reply'
        self.client.ctcpQuery_FINGER(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('FINGER', 'reply')]))])

    def test_ctcpQuery_SOURCE(self):
        """
        L{IRCClient.ctcpQuery_SOURCE} calls L{IRCClient.ctcpMakeReply} with the
        correct args.
        """
        self.client.sourceURL = 'url'
        self.client.ctcpQuery_SOURCE(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('SOURCE', 'url'), ('SOURCE', None)]))])

    def test_ctcpQuery_USERINFO(self):
        """
        L{IRCClient.ctcpQuery_USERINFO} calls L{IRCClient.ctcpMakeReply} with
        the correct args.
        """
        self.client.userinfo = 'info'
        self.client.ctcpQuery_USERINFO(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('USERINFO', 'info')]))])

    def test_ctcpQuery_CLIENTINFO(self):
        """
        L{IRCClient.ctcpQuery_CLIENTINFO} calls L{IRCClient.ctcpMakeReply} with
        the correct args.
        """
        self.client.ctcpQuery_CLIENTINFO(self.user, self.channel, '')
        self.client.ctcpQuery_CLIENTINFO(self.user, self.channel, 'PING PONG')
        info = 'ACTION CLIENTINFO DCC ERRMSG FINGER PING SOURCE TIME USERINFO VERSION'
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('CLIENTINFO', info)])), ('ctcpMakeReply', ('Wolf', [('CLIENTINFO', None)]))])

    def test_ctcpQuery_TIME(self):
        """
        L{IRCClient.ctcpQuery_TIME} calls L{IRCClient.ctcpMakeReply} with the
        correct args.
        """
        self.client.ctcpQuery_TIME(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods[0][1][0], 'Wolf')

    def test_ctcpQuery_DCC(self):
        """
        L{IRCClient.ctcpQuery_DCC} calls L{IRCClient.ctcpMakeReply} with the
        correct args.
        """
        self.client.ctcpQuery_DCC(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('ERRMSG', "DCC data :Unknown DCC type 'DATA'")]))])