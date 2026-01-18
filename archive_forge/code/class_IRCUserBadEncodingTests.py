from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
class IRCUserBadEncodingTests(IRCTestCase):
    """
    Verifies that L{IRCUser} sends the correct error messages back to clients
    when given indecipherable bytes
    """

    def setUp(self):
        self.ircUser = MocksyIRCUser()

    def assertChokesOnBadBytes(self, irc_x, error):
        """
        Asserts that IRCUser sends the relevant error code when a given irc_x
        dispatch method is given undecodable bytes.

        @param irc_x: the name of the irc_FOO method to test.
        For example, irc_x = 'PRIVMSG' will check irc_PRIVMSG

        @param error: the error code irc_x should send. For example,
        irc.ERR_NOTONCHANNEL
        """
        getattr(self.ircUser, 'irc_%s' % irc_x)(None, [BADTEXT])
        self.assertEqual(self.ircUser.mockedCodes, [error])

    def test_JOIN(self):
        """
        Tests that irc_JOIN sends ERR_NOSUCHCHANNEL if the channel name can't
        be decoded.
        """
        self.assertChokesOnBadBytes('JOIN', irc.ERR_NOSUCHCHANNEL)

    def test_NAMES(self):
        """
        Tests that irc_NAMES sends ERR_NOSUCHCHANNEL if the channel name can't
        be decoded.
        """
        self.assertChokesOnBadBytes('NAMES', irc.ERR_NOSUCHCHANNEL)

    def test_TOPIC(self):
        """
        Tests that irc_TOPIC sends ERR_NOSUCHCHANNEL if the channel name can't
        be decoded.
        """
        self.assertChokesOnBadBytes('TOPIC', irc.ERR_NOSUCHCHANNEL)

    def test_LIST(self):
        """
        Tests that irc_LIST sends ERR_NOSUCHCHANNEL if the channel name can't
        be decoded.
        """
        self.assertChokesOnBadBytes('LIST', irc.ERR_NOSUCHCHANNEL)

    def test_MODE(self):
        """
        Tests that irc_MODE sends ERR_NOSUCHNICK if the target name can't
        be decoded.
        """
        self.assertChokesOnBadBytes('MODE', irc.ERR_NOSUCHNICK)

    def test_PRIVMSG(self):
        """
        Tests that irc_PRIVMSG sends ERR_NOSUCHNICK if the target name can't
        be decoded.
        """
        self.assertChokesOnBadBytes('PRIVMSG', irc.ERR_NOSUCHNICK)

    def test_WHOIS(self):
        """
        Tests that irc_WHOIS sends ERR_NOSUCHNICK if the target name can't
        be decoded.
        """
        self.assertChokesOnBadBytes('WHOIS', irc.ERR_NOSUCHNICK)

    def test_PART(self):
        """
        Tests that irc_PART sends ERR_NOTONCHANNEL if the target name can't
        be decoded.
        """
        self.assertChokesOnBadBytes('PART', irc.ERR_NOTONCHANNEL)

    def test_WHO(self):
        """
        Tests that irc_WHO immediately ends the WHO list if the target name
        can't be decoded.
        """
        self.assertChokesOnBadBytes('WHO', irc.RPL_ENDOFWHO)