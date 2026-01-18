import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class DispatcherTests(IRCTestCase):
    """
    Tests for L{irc._CommandDispatcherMixin}.
    """

    def test_dispatch(self):
        """
        Dispatching a command invokes the correct handler.
        """
        disp = Dispatcher()
        args = (1, 2)
        res = disp.dispatch('working', *args)
        self.assertEqual(res, args)

    def test_dispatchUnknown(self):
        """
        Dispatching an unknown command invokes the default handler.
        """
        disp = Dispatcher()
        name = 'missing'
        args = (1, 2)
        res = disp.dispatch(name, *args)
        self.assertEqual(res, (name,) + args)

    def test_dispatchMissingUnknown(self):
        """
        Dispatching an unknown command, when no default handler is present,
        results in an exception being raised.
        """
        disp = Dispatcher()
        disp.disp_unknown = None
        self.assertRaises(irc.UnhandledCommand, disp.dispatch, 'bar')