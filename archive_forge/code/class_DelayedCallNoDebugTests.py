import socket
from queue import Queue
from typing import Callable
from unittest import skipIf
from zope.interface import implementer
from typing_extensions import ParamSpec
from twisted.internet._resolver import FirstOneWins
from twisted.internet.base import DelayedCall, ReactorBase, ThreadedResolver
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import IReactorThreads, IReactorTime, IResolverSimple
from twisted.internet.task import Clock
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import SkipTest, TestCase
class DelayedCallNoDebugTests(DelayedCallMixin, TestCase):
    """
    L{DelayedCall}
    """

    def setUp(self):
        """
        Turn debug off.
        """
        self.patch(DelayedCall, 'debug', False)
        DelayedCallMixin.setUp(self)

    def test_str(self):
        """
        The string representation of a L{DelayedCall} instance, as returned by
        L{str}, includes the unsigned id of the instance, as well as its state,
        the function to be called, and the function arguments.
        """
        dc = DelayedCall(12, nothing, (3,), {'A': 5}, None, None, lambda: 1.5)
        expected = '<DelayedCall 0x{:x} [10.5s] called=0 cancelled=0 nothing(3, A=5)>'.format(id(dc))
        self.assertEqual(str(dc), expected)

    def test_switchToDebug(self):
        """
        If L{DelayedCall.debug} changes from C{0} to C{1} between
        L{DelayeCall.__init__} and L{DelayedCall.__repr__} then
        L{DelayedCall.__repr__} returns a string that does not include the
        creator stack.
        """
        dc = DelayedCall(3, lambda: None, (), {}, nothing, nothing, lambda: 2)
        dc.debug = 1
        self.assertNotIn('traceback at creation', repr(dc))