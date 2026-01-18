from twisted.python import reflect, win32
from twisted.trial import unittest
class CommandLineQuotingTests(unittest.TestCase):
    """
    Tests for L{cmdLineQuote}.
    """

    def test_argWithoutSpaces(self) -> None:
        """
        Calling C{cmdLineQuote} with an argument with no spaces returns
        the argument unchanged.
        """
        self.assertEqual(win32.cmdLineQuote('an_argument'), 'an_argument')

    def test_argWithSpaces(self) -> None:
        """
        Calling C{cmdLineQuote} with an argument containing spaces returns
        the argument surrounded by quotes.
        """
        self.assertEqual(win32.cmdLineQuote('An Argument'), '"An Argument"')

    def test_emptyStringArg(self) -> None:
        """
        Calling C{cmdLineQuote} with an empty string returns a quoted empty
        string.
        """
        self.assertEqual(win32.cmdLineQuote(''), '""')