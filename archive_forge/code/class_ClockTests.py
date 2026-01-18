from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
class ClockTests(unittest.TestCase):
    """
    Test the non-wallclock based clock implementation.
    """

    def testSeconds(self):
        """
        Test that the C{seconds} method of the fake clock returns fake time.
        """
        c = task.Clock()
        self.assertEqual(c.seconds(), 0)

    def testCallLater(self):
        """
        Test that calls can be scheduled for later with the fake clock and
        hands back an L{IDelayedCall}.
        """
        c = task.Clock()
        call = c.callLater(1, lambda a, b: None, 1, b=2)
        self.assertTrue(interfaces.IDelayedCall.providedBy(call))
        self.assertEqual(call.getTime(), 1)
        self.assertTrue(call.active())

    def testCallLaterCancelled(self):
        """
        Test that calls can be cancelled.
        """
        c = task.Clock()
        call = c.callLater(1, lambda a, b: None, 1, b=2)
        call.cancel()
        self.assertFalse(call.active())

    def test_callLaterOrdering(self):
        """
        Test that the DelayedCall returned is not one previously
        created.
        """
        c = task.Clock()
        call1 = c.callLater(10, lambda a, b: None, 1, b=2)
        call2 = c.callLater(1, lambda a, b: None, 3, b=4)
        self.assertFalse(call1 is call2)

    def testAdvance(self):
        """
        Test that advancing the clock will fire some calls.
        """
        events = []
        c = task.Clock()
        call = c.callLater(2, lambda: events.append(None))
        c.advance(1)
        self.assertEqual(events, [])
        c.advance(1)
        self.assertEqual(events, [None])
        self.assertFalse(call.active())

    def testAdvanceCancel(self):
        """
        Test attempting to cancel the call in a callback.

        AlreadyCalled should be raised, not for example a ValueError from
        removing the call from Clock.calls. This requires call.called to be
        set before the callback is called.
        """
        c = task.Clock()

        def cb():
            self.assertRaises(error.AlreadyCalled, call.cancel)
        call = c.callLater(1, cb)
        c.advance(1)

    def testCallLaterDelayed(self):
        """
        Test that calls can be delayed.
        """
        events = []
        c = task.Clock()
        call = c.callLater(1, lambda a, b: events.append((a, b)), 1, b=2)
        call.delay(1)
        self.assertEqual(call.getTime(), 2)
        c.advance(1.5)
        self.assertEqual(events, [])
        c.advance(1.0)
        self.assertEqual(events, [(1, 2)])

    def testCallLaterResetLater(self):
        """
        Test that calls can have their time reset to a later time.
        """
        events = []
        c = task.Clock()
        call = c.callLater(2, lambda a, b: events.append((a, b)), 1, b=2)
        c.advance(1)
        call.reset(3)
        self.assertEqual(call.getTime(), 4)
        c.advance(2)
        self.assertEqual(events, [])
        c.advance(1)
        self.assertEqual(events, [(1, 2)])

    def testCallLaterResetSooner(self):
        """
        Test that calls can have their time reset to an earlier time.
        """
        events = []
        c = task.Clock()
        call = c.callLater(4, lambda a, b: events.append((a, b)), 1, b=2)
        call.reset(3)
        self.assertEqual(call.getTime(), 3)
        c.advance(3)
        self.assertEqual(events, [(1, 2)])

    def test_getDelayedCalls(self):
        """
        Test that we can get a list of all delayed calls
        """
        c = task.Clock()
        call = c.callLater(1, lambda x: None)
        call2 = c.callLater(2, lambda x: None)
        calls = c.getDelayedCalls()
        self.assertEqual({call, call2}, set(calls))

    def test_getDelayedCallsEmpty(self):
        """
        Test that we get an empty list from getDelayedCalls on a newly
        constructed Clock.
        """
        c = task.Clock()
        self.assertEqual(c.getDelayedCalls(), [])

    def test_providesIReactorTime(self):
        c = task.Clock()
        self.assertTrue(interfaces.IReactorTime.providedBy(c), 'Clock does not provide IReactorTime')

    def test_callLaterKeepsCallsOrdered(self):
        """
        The order of calls scheduled by L{task.Clock.callLater} is honored when
        adding a new call via calling L{task.Clock.callLater} again.

        For example, if L{task.Clock.callLater} is invoked with a callable "A"
        and a time t0, and then the L{IDelayedCall} which results from that is
        C{reset} to a later time t2 which is greater than t0, and I{then}
        L{task.Clock.callLater} is invoked again with a callable "B", and time
        t1 which is less than t2 but greater than t0, "B" will be invoked before
        "A".
        """
        result = []
        expected = [('b', 2.0), ('a', 3.0)]
        clock = task.Clock()
        logtime = lambda n: result.append((n, clock.seconds()))
        call_a = clock.callLater(1.0, logtime, 'a')
        call_a.reset(3.0)
        clock.callLater(2.0, logtime, 'b')
        clock.pump([1] * 3)
        self.assertEqual(result, expected)

    def test_callLaterResetKeepsCallsOrdered(self):
        """
        The order of calls scheduled by L{task.Clock.callLater} is honored when
        re-scheduling an existing call via L{IDelayedCall.reset} on the result
        of a previous call to C{callLater}.

        For example, if L{task.Clock.callLater} is invoked with a callable "A"
        and a time t0, and then L{task.Clock.callLater} is invoked again with a
        callable "B", and time t1 greater than t0, and finally the
        L{IDelayedCall} for "A" is C{reset} to a later time, t2, which is
        greater than t1, "B" will be invoked before "A".
        """
        result = []
        expected = [('b', 2.0), ('a', 3.0)]
        clock = task.Clock()
        logtime = lambda n: result.append((n, clock.seconds()))
        call_a = clock.callLater(1.0, logtime, 'a')
        clock.callLater(2.0, logtime, 'b')
        call_a.reset(3.0)
        clock.pump([1] * 3)
        self.assertEqual(result, expected)

    def test_callLaterResetInsideCallKeepsCallsOrdered(self):
        """
        The order of calls scheduled by L{task.Clock.callLater} is honored when
        re-scheduling an existing call via L{IDelayedCall.reset} on the result
        of a previous call to C{callLater}, even when that call to C{reset}
        occurs within the callable scheduled by C{callLater} itself.
        """
        result = []
        expected = [('c', 3.0), ('b', 4.0)]
        clock = task.Clock()
        logtime = lambda n: result.append((n, clock.seconds()))
        call_b = clock.callLater(2.0, logtime, 'b')

        def a():
            call_b.reset(3.0)
        clock.callLater(1.0, a)
        clock.callLater(3.0, logtime, 'c')
        clock.pump([0.5] * 10)
        self.assertEqual(result, expected)