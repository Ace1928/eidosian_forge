import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class MiscTests(IRCTestCase):
    """
    Tests for miscellaneous functions.
    """

    def test_foldr(self):
        """
        Apply a function of two arguments cumulatively to the items of
        a sequence, from right to left, so as to reduce the sequence to
        a single value.
        """
        self.assertEqual(irc._foldr(operator.sub, 0, [1, 2, 3, 4]), -2)

        def insertTop(l, x):
            l.insert(0, x)
            return l
        self.assertEqual(irc._foldr(insertTop, [], [[1], [2], [3], [4]]), [[[[[], 4], 3], 2], 1])