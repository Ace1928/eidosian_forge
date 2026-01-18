import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class CTCPTests(IRCTestCase):
    """
    Tests for L{twisted.words.protocols.irc.IRCClient} CTCP handling.
    """

    def setUp(self):
        self.file = StringIOWithoutClosing()
        self.transport = protocol.FileWrapper(self.file)
        self.client = IRCClientWithoutLogin()
        self.client.makeConnection(self.transport)
        self.addCleanup(self.transport.loseConnection)
        self.addCleanup(self.client.connectionLost, None)

    def test_ERRMSG(self):
        """Testing CTCP query ERRMSG.

        Not because this is this is an especially important case in the
        field, but it does go through the entire dispatch/decode/encode
        process.
        """
        errQuery = ':nick!guy@over.there PRIVMSG #theChan :%(X)cERRMSG t%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF}
        errReply = 'NOTICE nick :%(X)cERRMSG t :No error has occurred.%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF}
        self.client.dataReceived(errQuery)
        reply = self.file.getvalue()
        self.assertEqualBufferValue(reply, errReply)

    def test_noNumbersVERSION(self):
        """
        If attributes for version information on L{IRCClient} are set to
        L{None}, the parts of the CTCP VERSION response they correspond to
        are omitted.
        """
        self.client.versionName = 'FrobozzIRC'
        self.client.ctcpQuery_VERSION('nick!guy@over.there', '#theChan', None)
        versionReply = 'NOTICE nick :%(X)cVERSION %(vname)s::%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF, 'vname': self.client.versionName}
        reply = self.file.getvalue()
        self.assertEqualBufferValue(reply, versionReply)

    def test_fullVERSION(self):
        """
        The response to a CTCP VERSION query includes the version number and
        environment information, as specified by L{IRCClient.versionNum} and
        L{IRCClient.versionEnv}.
        """
        self.client.versionName = 'FrobozzIRC'
        self.client.versionNum = '1.2g'
        self.client.versionEnv = 'ZorkOS'
        self.client.ctcpQuery_VERSION('nick!guy@over.there', '#theChan', None)
        versionReply = 'NOTICE nick :%(X)cVERSION %(vname)s:%(vnum)s:%(venv)s%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF, 'vname': self.client.versionName, 'vnum': self.client.versionNum, 'venv': self.client.versionEnv}
        reply = self.file.getvalue()
        self.assertEqualBufferValue(reply, versionReply)

    def test_noDuplicateCTCPDispatch(self):
        """
        Duplicated CTCP messages are ignored and no reply is made.
        """

        def testCTCP(user, channel, data):
            self.called += 1
        self.called = 0
        self.client.ctcpQuery_TESTTHIS = testCTCP
        self.client.irc_PRIVMSG('foo!bar@baz.quux', ['#chan', '{X}TESTTHIS{X}foo{X}TESTTHIS{X}'.format(X=irc.X_DELIM)])
        self.assertEqualBufferValue(self.file.getvalue(), '')
        self.assertEqual(self.called, 1)

    def test_noDefaultDispatch(self):
        """
        The fallback handler is invoked for unrecognized CTCP messages.
        """

        def unknownQuery(user, channel, tag, data):
            self.calledWith = (user, channel, tag, data)
            self.called += 1
        self.called = 0
        self.patch(self.client, 'ctcpUnknownQuery', unknownQuery)
        self.client.irc_PRIVMSG('foo!bar@baz.quux', ['#chan', '{X}NOTREAL{X}'.format(X=irc.X_DELIM)])
        self.assertEqualBufferValue(self.file.getvalue(), '')
        self.assertEqual(self.calledWith, ('foo!bar@baz.quux', '#chan', 'NOTREAL', None))
        self.assertEqual(self.called, 1)
        self.client.irc_PRIVMSG('foo!bar@baz.quux', ['#chan', '{X}NOTREAL{X}foo{X}NOTREAL{X}'.format(X=irc.X_DELIM)])
        self.assertEqual(self.called, 2)