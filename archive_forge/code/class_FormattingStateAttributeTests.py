import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class FormattingStateAttributeTests(IRCTestCase):
    """
    Tests for L{twisted.words.protocols.irc._FormattingState}.
    """

    def test_equality(self):
        """
        L{irc._FormattingState}s must have matching character attribute
        values (bold, underline, etc) with the same values to be considered
        equal.
        """
        self.assertEqual(irc._FormattingState(), irc._FormattingState())
        self.assertEqual(irc._FormattingState(), irc._FormattingState(off=False))
        self.assertEqual(irc._FormattingState(bold=True, underline=True, off=False, reverseVideo=True, foreground=irc._IRC_COLORS['blue']), irc._FormattingState(bold=True, underline=True, off=False, reverseVideo=True, foreground=irc._IRC_COLORS['blue']))
        self.assertNotEqual(irc._FormattingState(bold=True), irc._FormattingState(bold=False))