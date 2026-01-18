from twisted.internet import defer, reactor, task
from twisted.trial import unittest
class CooperatorTests(unittest.TestCase):
    RESULT = 'done'

    def ebIter(self, err):
        err.trap(task.SchedulerStopped)
        return self.RESULT

    def cbIter(self, ign):
        self.fail()

    def testStoppedRejectsNewTasks(self):
        """
        Test that Cooperators refuse new tasks when they have been stopped.
        """

        def testwith(stuff):
            c = task.Cooperator()
            c.stop()
            d = c.coiterate(iter(()), stuff)
            d.addCallback(self.cbIter)
            d.addErrback(self.ebIter)
            return d.addCallback(lambda result: self.assertEqual(result, self.RESULT))
        return testwith(None).addCallback(lambda ign: testwith(defer.Deferred()))

    def testStopRunning(self):
        """
        Test that a running iterator will not run to completion when the
        cooperator is stopped.
        """
        c = task.Cooperator()

        def myiter():
            yield from range(3)
        myiter.value = -1
        d = c.coiterate(myiter())
        d.addCallback(self.cbIter)
        d.addErrback(self.ebIter)
        c.stop()

        def doasserts(result):
            self.assertEqual(result, self.RESULT)
            self.assertEqual(myiter.value, -1)
        d.addCallback(doasserts)
        return d

    def testStopOutstanding(self):
        """
        An iterator run with L{Cooperator.coiterate} paused on a L{Deferred}
        yielded by that iterator will fire its own L{Deferred} (the one
        returned by C{coiterate}) when L{Cooperator.stop} is called.
        """
        testControlD = defer.Deferred()
        outstandingD = defer.Deferred()

        def myiter():
            reactor.callLater(0, testControlD.callback, None)
            yield outstandingD
            self.fail()
        c = task.Cooperator()
        d = c.coiterate(myiter())

        def stopAndGo(ign):
            c.stop()
            outstandingD.callback('arglebargle')
        testControlD.addCallback(stopAndGo)
        d.addCallback(self.cbIter)
        d.addErrback(self.ebIter)
        return d.addCallback(lambda result: self.assertEqual(result, self.RESULT))

    def testUnexpectedError(self):
        c = task.Cooperator()

        def myiter():
            if False:
                yield None
            else:
                raise RuntimeError()
        d = c.coiterate(myiter())
        return self.assertFailure(d, RuntimeError)

    def testUnexpectedErrorActuallyLater(self):

        def myiter():
            D = defer.Deferred()
            reactor.callLater(0, D.errback, RuntimeError())
            yield D
        c = task.Cooperator()
        d = c.coiterate(myiter())
        return self.assertFailure(d, RuntimeError)

    def testUnexpectedErrorNotActuallyLater(self):

        def myiter():
            yield defer.fail(RuntimeError())
        c = task.Cooperator()
        d = c.coiterate(myiter())
        return self.assertFailure(d, RuntimeError)

    def testCooperation(self):
        L = []

        def myiter(things):
            for th in things:
                L.append(th)
                yield None
        groupsOfThings = ['abc', (1, 2, 3), 'def', (4, 5, 6)]
        c = task.Cooperator()
        tasks = []
        for stuff in groupsOfThings:
            tasks.append(c.coiterate(myiter(stuff)))
        return defer.DeferredList(tasks).addCallback(lambda ign: self.assertEqual(tuple(L), sum(zip(*groupsOfThings), ())))

    def testResourceExhaustion(self):
        output = []

        def myiter():
            for i in range(100):
                output.append(i)
                if i == 9:
                    _TPF.stopped = True
                yield i

        class _TPF:
            stopped = False

            def __call__(self):
                return self.stopped
        c = task.Cooperator(terminationPredicateFactory=_TPF)
        c.coiterate(myiter()).addErrback(self.ebIter)
        c._delayedCall.cancel()
        c._tick()
        c.stop()
        self.assertTrue(_TPF.stopped)
        self.assertEqual(output, list(range(10)))

    def testCallbackReCoiterate(self):
        """
        If a callback to a deferred returned by coiterate calls coiterate on
        the same Cooperator, we should make sure to only do the minimal amount
        of scheduling work.  (This test was added to demonstrate a specific bug
        that was found while writing the scheduler.)
        """
        calls = []

        class FakeCall:

            def __init__(self, func):
                self.func = func

            def __repr__(self) -> str:
                return f'<FakeCall {self.func!r}>'

        def sched(f):
            self.assertFalse(calls, repr(calls))
            calls.append(FakeCall(f))
            return calls[-1]
        c = task.Cooperator(scheduler=sched, terminationPredicateFactory=lambda: lambda: True)
        d = c.coiterate(iter(()))
        done = []

        def anotherTask(ign):
            c.coiterate(iter(())).addBoth(done.append)
        d.addCallback(anotherTask)
        work = 0
        while not done:
            work += 1
            while calls:
                calls.pop(0).func()
                work += 1
            if work > 50:
                self.fail('Cooperator took too long')

    def test_removingLastTaskStopsScheduledCall(self):
        """
        If the last task in a Cooperator is removed, the scheduled call for
        the next tick is cancelled, since it is no longer necessary.

        This behavior is useful for tests that want to assert they have left
        no reactor state behind when they're done.
        """
        calls = [None]

        def sched(f):
            calls[0] = FakeDelayedCall(f)
            return calls[0]
        coop = task.Cooperator(scheduler=sched)
        task1 = coop.cooperate(iter([1, 2]))
        task2 = coop.cooperate(iter([1, 2]))
        self.assertEqual(calls[0].func, coop._tick)
        task1.stop()
        self.assertFalse(calls[0].cancelled)
        self.assertEqual(coop._delayedCall, calls[0])
        task2.stop()
        self.assertTrue(calls[0].cancelled)
        self.assertIsNone(coop._delayedCall)
        coop.cooperate(iter([1, 2]))
        self.assertFalse(calls[0].cancelled)
        self.assertEqual(coop._delayedCall, calls[0])

    def test_runningWhenStarted(self):
        """
        L{Cooperator.running} reports C{True} if the L{Cooperator}
        was started on creation.
        """
        c = task.Cooperator()
        self.assertTrue(c.running)

    def test_runningWhenNotStarted(self):
        """
        L{Cooperator.running} reports C{False} if the L{Cooperator}
        has not been started.
        """
        c = task.Cooperator(started=False)
        self.assertFalse(c.running)

    def test_runningWhenRunning(self):
        """
        L{Cooperator.running} reports C{True} when the L{Cooperator}
        is running.
        """
        c = task.Cooperator(started=False)
        c.start()
        self.addCleanup(c.stop)
        self.assertTrue(c.running)

    def test_runningWhenStopped(self):
        """
        L{Cooperator.running} reports C{False} after the L{Cooperator}
        has been stopped.
        """
        c = task.Cooperator(started=False)
        c.start()
        c.stop()
        self.assertFalse(c.running)