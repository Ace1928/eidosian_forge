import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class ClientImplementationTests(IRCTestCase):

    def setUp(self):
        self.transport = StringTransport()
        self.client = NoticingClient()
        self.client.makeConnection(self.transport)
        self.addCleanup(self.transport.loseConnection)
        self.addCleanup(self.client.connectionLost, None)

    def _serverTestImpl(self, code, msg, func, **kw):
        host = pop(kw, 'host', 'server.host')
        nick = pop(kw, 'nick', 'nickname')
        args = pop(kw, 'args', '')
        message = ':' + host + ' ' + code + ' ' + nick + ' ' + args + ' :' + msg + '\r\n'
        self.client.dataReceived(message)
        self.assertEqual(self.client.calls, [(func, kw)])

    def testYourHost(self):
        msg = 'Your host is some.host[blah.blah/6667], running version server-version-3'
        self._serverTestImpl('002', msg, 'yourHost', info=msg)

    def testCreated(self):
        msg = 'This server was cobbled together Fri Aug 13 18:00:25 UTC 2004'
        self._serverTestImpl('003', msg, 'created', when=msg)

    def testMyInfo(self):
        msg = 'server.host server-version abcDEF bcdEHI'
        self._serverTestImpl('004', msg, 'myInfo', servername='server.host', version='server-version', umodes='abcDEF', cmodes='bcdEHI')

    def testLuserClient(self):
        msg = 'There are 9227 victims and 9542 hiding on 24 servers'
        self._serverTestImpl('251', msg, 'luserClient', info=msg)

    def _sendISUPPORT(self):
        args = 'MODES=4 CHANLIMIT=#:20 NICKLEN=16 USERLEN=10 HOSTLEN=63 TOPICLEN=450 KICKLEN=450 CHANNELLEN=30 KEYLEN=23 CHANTYPES=# PREFIX=(ov)@+ CASEMAPPING=ascii CAPAB IRCD=dancer'
        msg = 'are available on this server'
        self._serverTestImpl('005', msg, 'isupport', args=args, options=['MODES=4', 'CHANLIMIT=#:20', 'NICKLEN=16', 'USERLEN=10', 'HOSTLEN=63', 'TOPICLEN=450', 'KICKLEN=450', 'CHANNELLEN=30', 'KEYLEN=23', 'CHANTYPES=#', 'PREFIX=(ov)@+', 'CASEMAPPING=ascii', 'CAPAB', 'IRCD=dancer'])

    def test_ISUPPORT(self):
        """
        The client parses ISUPPORT messages sent by the server and calls
        L{IRCClient.isupport}.
        """
        self._sendISUPPORT()

    def testBounce(self):
        msg = 'Try server some.host, port 321'
        self._serverTestImpl('010', msg, 'bounce', info=msg)

    def testLuserChannels(self):
        args = '7116'
        msg = 'channels formed'
        self._serverTestImpl('254', msg, 'luserChannels', args=args, channels=int(args))

    def testLuserOp(self):
        args = '34'
        msg = 'flagged staff members'
        self._serverTestImpl('252', msg, 'luserOp', args=args, ops=int(args))

    def testLuserMe(self):
        msg = 'I have 1937 clients and 0 servers'
        self._serverTestImpl('255', msg, 'luserMe', info=msg)

    def test_receivedMOTD(self):
        """
        Lines received in I{RPL_MOTDSTART} and I{RPL_MOTD} are delivered to
        L{IRCClient.receivedMOTD} when I{RPL_ENDOFMOTD} is received.
        """
        lines = [':host.name 375 nickname :- host.name Message of the Day -', ':host.name 372 nickname :- Welcome to host.name', ':host.name 376 nickname :End of /MOTD command.']
        for L in lines:
            self.assertEqual(self.client.calls, [])
            self.client.dataReceived(L + '\r\n')
        self.assertEqual(self.client.calls, [('receivedMOTD', {'motd': ['host.name Message of the Day -', 'Welcome to host.name']})])
        self.assertIdentical(self.client.motd, None)

    def test_withoutMOTDSTART(self):
        """
        If L{IRCClient} receives I{RPL_MOTD} and I{RPL_ENDOFMOTD} without
        receiving I{RPL_MOTDSTART}, L{IRCClient.receivedMOTD} is still
        called with a list of MOTD lines.
        """
        lines = [':host.name 372 nickname :- Welcome to host.name', ':host.name 376 nickname :End of /MOTD command.']
        for L in lines:
            self.client.dataReceived(L + '\r\n')
        self.assertEqual(self.client.calls, [('receivedMOTD', {'motd': ['Welcome to host.name']})])

    def _clientTestImpl(self, sender, group, type, msg, func, **kw):
        ident = pop(kw, 'ident', 'ident')
        host = pop(kw, 'host', 'host')
        wholeUser = sender + '!' + ident + '@' + host
        message = ':' + wholeUser + ' ' + type + ' ' + group + ' :' + msg + '\r\n'
        self.client.dataReceived(message)
        self.assertEqual(self.client.calls, [(func, kw)])
        self.client.calls = []

    def testPrivmsg(self):
        msg = 'Tooty toot toot.'
        self._clientTestImpl('sender', '#group', 'PRIVMSG', msg, 'privmsg', ident='ident', host='host', user='sender!ident@host', channel='#group', message=msg)
        self._clientTestImpl('sender', 'recipient', 'PRIVMSG', msg, 'privmsg', ident='ident', host='host', user='sender!ident@host', channel='recipient', message=msg)

    def test_getChannelModeParams(self):
        """
        L{IRCClient.getChannelModeParams} uses ISUPPORT information, either
        given by the server or defaults, to determine which channel modes
        require arguments when being added or removed.
        """
        add, remove = map(sorted, self.client.getChannelModeParams())
        self.assertEqual(add, ['b', 'h', 'k', 'l', 'o', 'v'])
        self.assertEqual(remove, ['b', 'h', 'o', 'v'])

        def removeFeature(name):
            name = '-' + name
            msg = 'are available on this server'
            self._serverTestImpl('005', msg, 'isupport', args=name, options=[name])
            self.assertIdentical(self.client.supported.getFeature(name), None)
            self.client.calls = []
        removeFeature('CHANMODES')
        add, remove = map(sorted, self.client.getChannelModeParams())
        self.assertEqual(add, ['h', 'o', 'v'])
        self.assertEqual(remove, ['h', 'o', 'v'])
        removeFeature('PREFIX')
        add, remove = map(sorted, self.client.getChannelModeParams())
        self.assertEqual(add, [])
        self.assertEqual(remove, [])
        self._sendISUPPORT()
        self.assertNotIdentical(self.client.supported.getFeature('PREFIX'), None)

    def test_getUserModeParams(self):
        """
        L{IRCClient.getUserModeParams} returns a list of user modes (modes that
        the user sets on themself, outside of channel modes) that require
        parameters when added and removed, respectively.
        """
        add, remove = map(sorted, self.client.getUserModeParams())
        self.assertEqual(add, [])
        self.assertEqual(remove, [])

    def _sendModeChange(self, msg, args='', target=None):
        """
        Build a MODE string and send it to the client.
        """
        if target is None:
            target = '#chan'
        message = f':Wolf!~wolf@yok.utu.fi MODE {target} {msg} {args}\r\n'
        self.client.dataReceived(message)

    def _parseModeChange(self, results, target=None):
        """
        Parse the results, do some test and return the data to check.
        """
        if target is None:
            target = '#chan'
        for n, result in enumerate(results):
            method, data = result
            self.assertEqual(method, 'modeChanged')
            self.assertEqual(data['user'], 'Wolf!~wolf@yok.utu.fi')
            self.assertEqual(data['channel'], target)
            results[n] = tuple((data[key] for key in ('set', 'modes', 'args')))
        return results

    def _checkModeChange(self, expected, target=None):
        """
        Compare the expected result with the one returned by the client.
        """
        result = self._parseModeChange(self.client.calls, target)
        self.assertEqual(result, expected)
        self.client.calls = []

    def test_modeMissingDirection(self):
        """
        Mode strings that do not begin with a directional character, C{'+'} or
        C{'-'}, have C{'+'} automatically prepended.
        """
        self._sendModeChange('s')
        self._checkModeChange([(True, 's', (None,))])

    def test_noModeParameters(self):
        """
        No parameters are passed to L{IRCClient.modeChanged} for modes that
        don't take any parameters.
        """
        self._sendModeChange('-s')
        self._checkModeChange([(False, 's', (None,))])
        self._sendModeChange('+n')
        self._checkModeChange([(True, 'n', (None,))])

    def test_oneModeParameter(self):
        """
        Parameters are passed to L{IRCClient.modeChanged} for modes that take
        parameters.
        """
        self._sendModeChange('+o', 'a_user')
        self._checkModeChange([(True, 'o', ('a_user',))])
        self._sendModeChange('-o', 'a_user')
        self._checkModeChange([(False, 'o', ('a_user',))])

    def test_mixedModes(self):
        """
        Mixing adding and removing modes that do and don't take parameters
        invokes L{IRCClient.modeChanged} with mode characters and parameters
        that match up.
        """
        self._sendModeChange('+osv', 'a_user another_user')
        self._checkModeChange([(True, 'osv', ('a_user', None, 'another_user'))])
        self._sendModeChange('+v-os', 'a_user another_user')
        self._checkModeChange([(True, 'v', ('a_user',)), (False, 'os', ('another_user', None))])

    def test_tooManyModeParameters(self):
        """
        Passing an argument to modes that take no parameters results in
        L{IRCClient.modeChanged} not being called and an error being logged.
        """
        self._sendModeChange('+s', 'wrong')
        self._checkModeChange([])
        errors = self.flushLoggedErrors(irc.IRCBadModes)
        self.assertEqual(len(errors), 1)
        self.assertSubstring('Too many parameters', errors[0].getErrorMessage())

    def test_tooFewModeParameters(self):
        """
        Passing no arguments to modes that do take parameters results in
        L{IRCClient.modeChange} not being called and an error being logged.
        """
        self._sendModeChange('+o')
        self._checkModeChange([])
        errors = self.flushLoggedErrors(irc.IRCBadModes)
        self.assertEqual(len(errors), 1)
        self.assertSubstring('Not enough parameters', errors[0].getErrorMessage())

    def test_userMode(self):
        """
        A C{MODE} message whose target is our user (the nickname of our user,
        to be precise), as opposed to a channel, will be parsed according to
        the modes specified by L{IRCClient.getUserModeParams}.
        """
        target = self.client.nickname
        self._sendModeChange('+o', target=target)
        self._checkModeChange([(True, 'o', (None,))], target=target)

        def getUserModeParams():
            return ['Z', '']
        self.patch(self.client, 'getUserModeParams', getUserModeParams)
        self._sendModeChange('+Z', 'an_arg', target=target)
        self._checkModeChange([(True, 'Z', ('an_arg',))], target=target)

    def test_heartbeat(self):
        """
        When the I{RPL_WELCOME} message is received a heartbeat is started that
        will send a I{PING} message to the IRC server every
        L{irc.IRCClient.heartbeatInterval} seconds. When the transport is
        closed the heartbeat looping call is stopped too.
        """

        def _createHeartbeat():
            heartbeat = self._originalCreateHeartbeat()
            heartbeat.clock = self.clock
            return heartbeat
        self.clock = task.Clock()
        self._originalCreateHeartbeat = self.client._createHeartbeat
        self.patch(self.client, '_createHeartbeat', _createHeartbeat)
        self.assertIdentical(self.client._heartbeat, None)
        self.client.irc_RPL_WELCOME('foo', [])
        self.assertNotIdentical(self.client._heartbeat, None)
        self.assertEqual(self.client.hostname, 'foo')
        self.assertEqualBufferValue(self.transport.value(), '')
        self.clock.advance(self.client.heartbeatInterval)
        self.assertEqualBufferValue(self.transport.value(), 'PING foo\r\n')
        self.transport.loseConnection()
        self.client.connectionLost(None)
        self.assertEqual(len(self.clock.getDelayedCalls()), 0)
        self.assertIdentical(self.client._heartbeat, None)

    def test_heartbeatDisabled(self):
        """
        If L{irc.IRCClient.heartbeatInterval} is set to L{None} then no
        heartbeat is created.
        """
        self.assertIdentical(self.client._heartbeat, None)
        self.client.heartbeatInterval = None
        self.client.irc_RPL_WELCOME('foo', [])
        self.assertIdentical(self.client._heartbeat, None)