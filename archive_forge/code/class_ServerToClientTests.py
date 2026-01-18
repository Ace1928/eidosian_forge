import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class ServerToClientTests(IRCTestCase):
    """
    Tests for the C{irc_*} methods sent from the server to the client.
    """

    def setUp(self):
        self.user = 'Wolf!~wolf@yok.utu.fi'
        self.channel = '#twisted'
        methods = ['joined', 'userJoined', 'left', 'userLeft', 'userQuit', 'noticed', 'kickedFrom', 'userKicked', 'topicUpdated']
        self.client = CollectorClient(methods)

    def test_irc_JOIN(self):
        """
        L{IRCClient.joined} is called when I join a channel;
        L{IRCClient.userJoined} is called when someone else joins.
        """
        self.client.irc_JOIN(self.user, [self.channel])
        self.client.irc_JOIN('Svadilfari!~svadi@yok.utu.fi', ['#python'])
        self.assertEqual(self.client.methods, [('joined', (self.channel,)), ('userJoined', ('Svadilfari', '#python'))])

    def test_irc_PART(self):
        """
        L{IRCClient.left} is called when I part the channel;
        L{IRCClient.userLeft} is called when someone else parts.
        """
        self.client.irc_PART(self.user, [self.channel])
        self.client.irc_PART('Svadilfari!~svadi@yok.utu.fi', ['#python'])
        self.assertEqual(self.client.methods, [('left', (self.channel,)), ('userLeft', ('Svadilfari', '#python'))])

    def test_irc_QUIT(self):
        """
        L{IRCClient.userQuit} is called whenever someone quits
        the channel (myself included).
        """
        self.client.irc_QUIT('Svadilfari!~svadi@yok.utu.fi', ['Adios.'])
        self.client.irc_QUIT(self.user, ['Farewell.'])
        self.assertEqual(self.client.methods, [('userQuit', ('Svadilfari', 'Adios.')), ('userQuit', ('Wolf', 'Farewell.'))])

    def test_irc_NOTICE(self):
        """
        L{IRCClient.noticed} is called when a notice is received.
        """
        msg = '%(X)cextended%(X)cdata1%(X)cextended%(X)cdata2%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF}
        self.client.irc_NOTICE(self.user, [self.channel, msg])
        self.assertEqual(self.client.methods, [('noticed', (self.user, '#twisted', 'data1 data2'))])

    def test_irc_KICK(self):
        """
        L{IRCClient.kickedFrom} is called when I get kicked from the channel;
        L{IRCClient.userKicked} is called when someone else gets kicked.
        """
        self.client.irc_KICK('Svadilfari!~svadi@yok.utu.fi', ['#python', 'WOLF', 'shoryuken!'])
        self.client.irc_KICK(self.user, [self.channel, 'Svadilfari', 'hadouken!'])
        self.assertEqual(self.client.methods, [('kickedFrom', ('#python', 'Svadilfari', 'shoryuken!')), ('userKicked', ('Svadilfari', self.channel, 'Wolf', 'hadouken!'))])

    def test_irc_TOPIC(self):
        """
        L{IRCClient.topicUpdated} is called when someone sets the topic.
        """
        self.client.irc_TOPIC(self.user, [self.channel, 'new topic is new'])
        self.assertEqual(self.client.methods, [('topicUpdated', ('Wolf', self.channel, 'new topic is new'))])

    def test_irc_RPL_TOPIC(self):
        """
        L{IRCClient.topicUpdated} is called when the topic is initially
        reported.
        """
        self.client.irc_RPL_TOPIC(self.user, ['?', self.channel, 'new topic is new'])
        self.assertEqual(self.client.methods, [('topicUpdated', ('Wolf', self.channel, 'new topic is new'))])

    def test_irc_RPL_NOTOPIC(self):
        """
        L{IRCClient.topicUpdated} is called when the topic is removed.
        """
        self.client.irc_RPL_NOTOPIC(self.user, ['?', self.channel])
        self.assertEqual(self.client.methods, [('topicUpdated', ('Wolf', self.channel, ''))])