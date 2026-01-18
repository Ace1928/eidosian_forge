from twisted.internet import defer, reactor, task
from twisted.trial import unittest
class RunStateTests(unittest.TestCase):
    """
    Tests to verify the behavior of L{CooperativeTask.pause},
    L{CooperativeTask.resume}, L{CooperativeTask.stop}, exhausting the
    underlying iterator, and their interactions with each other.
    """

    def setUp(self):
        """
        Create a cooperator with a fake scheduler and a termination predicate
        that ensures only one unit of work will take place per tick.
        """
        self._doDeferNext = False
        self._doStopNext = False
        self._doDieNext = False
        self.work = []
        self.scheduler = FakeScheduler()
        self.cooperator = task.Cooperator(scheduler=self.scheduler, terminationPredicateFactory=lambda: lambda: True)
        self.task = self.cooperator.cooperate(self.worker())
        self.cooperator.start()

    def worker(self):
        """
        This is a sample generator which yields Deferreds when we are testing
        deferral and an ascending integer count otherwise.
        """
        i = 0
        while True:
            i += 1
            if self._doDeferNext:
                self._doDeferNext = False
                d = defer.Deferred()
                self.work.append(d)
                yield d
            elif self._doStopNext:
                return
            elif self._doDieNext:
                raise UnhandledException()
            else:
                self.work.append(i)
                yield i

    def tearDown(self):
        """
        Drop references to interesting parts of the fixture to allow Deferred
        errors to be noticed when things start failing.
        """
        del self.task
        del self.scheduler

    def deferNext(self):
        """
        Defer the next result from my worker iterator.
        """
        self._doDeferNext = True

    def stopNext(self):
        """
        Make the next result from my worker iterator be completion (raising
        StopIteration).
        """
        self._doStopNext = True

    def dieNext(self):
        """
        Make the next result from my worker iterator be raising an
        L{UnhandledException}.
        """

        def ignoreUnhandled(failure):
            failure.trap(UnhandledException)
            return None
        self._doDieNext = True

    def test_pauseResume(self):
        """
        Cooperators should stop running their tasks when they're paused, and
        start again when they're resumed.
        """
        self.scheduler.pump()
        self.assertEqual(self.work, [1])
        self.scheduler.pump()
        self.assertEqual(self.work, [1, 2])
        self.task.pause()
        self.scheduler.pump()
        self.assertEqual(self.work, [1, 2])
        self.task.resume()
        self.assertEqual(self.work, [1, 2])
        self.scheduler.pump()
        self.assertEqual(self.work, [1, 2, 3])

    def test_resumeNotPaused(self):
        """
        L{CooperativeTask.resume} should raise a L{TaskNotPaused} exception if
        it was not paused; e.g. if L{CooperativeTask.pause} was not invoked
        more times than L{CooperativeTask.resume} on that object.
        """
        self.assertRaises(task.NotPaused, self.task.resume)
        self.task.pause()
        self.task.resume()
        self.assertRaises(task.NotPaused, self.task.resume)

    def test_pauseTwice(self):
        """
        Pauses on tasks should behave like a stack. If a task is paused twice,
        it needs to be resumed twice.
        """
        self.task.pause()
        self.scheduler.pump()
        self.assertEqual(self.work, [])
        self.task.pause()
        self.scheduler.pump()
        self.assertEqual(self.work, [])
        self.task.resume()
        self.scheduler.pump()
        self.assertEqual(self.work, [])
        self.task.resume()
        self.scheduler.pump()
        self.assertEqual(self.work, [1])

    def test_pauseWhileDeferred(self):
        """
        C{pause()}ing a task while it is waiting on an outstanding
        L{defer.Deferred} should put the task into a state where the
        outstanding L{defer.Deferred} must be called back I{and} the task is
        C{resume}d before it will continue processing.
        """
        self.deferNext()
        self.scheduler.pump()
        self.assertEqual(len(self.work), 1)
        self.assertIsInstance(self.work[0], defer.Deferred)
        self.scheduler.pump()
        self.assertEqual(len(self.work), 1)
        self.task.pause()
        self.scheduler.pump()
        self.assertEqual(len(self.work), 1)
        self.task.resume()
        self.scheduler.pump()
        self.assertEqual(len(self.work), 1)
        self.work[0].callback('STUFF!')
        self.scheduler.pump()
        self.assertEqual(len(self.work), 2)
        self.assertEqual(self.work[1], 2)

    def test_whenDone(self):
        """
        L{CooperativeTask.whenDone} returns a Deferred which fires when the
        Cooperator's iterator is exhausted.  It returns a new Deferred each
        time it is called; callbacks added to other invocations will not modify
        the value that subsequent invocations will fire with.
        """
        deferred1 = self.task.whenDone()
        deferred2 = self.task.whenDone()
        results1 = []
        results2 = []
        final1 = []
        final2 = []

        def callbackOne(result):
            results1.append(result)
            return 1

        def callbackTwo(result):
            results2.append(result)
            return 2
        deferred1.addCallback(callbackOne)
        deferred2.addCallback(callbackTwo)
        deferred1.addCallback(final1.append)
        deferred2.addCallback(final2.append)
        self.stopNext()
        self.scheduler.pump()
        self.assertEqual(len(results1), 1)
        self.assertEqual(len(results2), 1)
        self.assertIs(results1[0], self.task._iterator)
        self.assertIs(results2[0], self.task._iterator)
        self.assertEqual(final1, [1])
        self.assertEqual(final2, [2])

    def test_whenDoneError(self):
        """
        L{CooperativeTask.whenDone} returns a L{defer.Deferred} that will fail
        when the iterable's C{next} method raises an exception, with that
        exception.
        """
        deferred1 = self.task.whenDone()
        results = []
        deferred1.addErrback(results.append)
        self.dieNext()
        self.scheduler.pump()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].check(UnhandledException), UnhandledException)

    def test_whenDoneStop(self):
        """
        L{CooperativeTask.whenDone} returns a L{defer.Deferred} that fails with
        L{TaskStopped} when the C{stop} method is called on that
        L{CooperativeTask}.
        """
        deferred1 = self.task.whenDone()
        errors = []
        deferred1.addErrback(errors.append)
        self.task.stop()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].check(task.TaskStopped), task.TaskStopped)

    def test_whenDoneAlreadyDone(self):
        """
        L{CooperativeTask.whenDone} will return a L{defer.Deferred} that will
        succeed immediately if its iterator has already completed.
        """
        self.stopNext()
        self.scheduler.pump()
        results = []
        self.task.whenDone().addCallback(results.append)
        self.assertEqual(results, [self.task._iterator])

    def test_stopStops(self):
        """
        C{stop()}ping a task should cause it to be removed from the run just as
        C{pause()}ing, with the distinction that C{resume()} will raise a
        L{TaskStopped} exception.
        """
        self.task.stop()
        self.scheduler.pump()
        self.assertEqual(len(self.work), 0)
        self.assertRaises(task.TaskStopped, self.task.stop)
        self.assertRaises(task.TaskStopped, self.task.pause)
        self.scheduler.pump()
        self.assertEqual(self.work, [])

    def test_pauseStopResume(self):
        """
        C{resume()}ing a paused, stopped task should be a no-op; it should not
        raise an exception, because it's paused, but neither should it actually
        do more work from the task.
        """
        self.task.pause()
        self.task.stop()
        self.task.resume()
        self.scheduler.pump()
        self.assertEqual(self.work, [])

    def test_stopDeferred(self):
        """
        As a corrolary of the interaction of C{pause()} and C{unpause()},
        C{stop()}ping a task which is waiting on a L{Deferred} should cause the
        task to gracefully shut down, meaning that it should not be unpaused
        when the deferred fires.
        """
        self.deferNext()
        self.scheduler.pump()
        d = self.work.pop()
        self.assertEqual(self.task._pauseCount, 1)
        results = []
        d.addBoth(results.append)
        self.scheduler.pump()
        self.task.stop()
        self.scheduler.pump()
        d.callback(7)
        self.scheduler.pump()
        self.assertEqual(results, [None])
        self.assertEqual(self.work, [])

    def test_stopExhausted(self):
        """
        C{stop()}ping a L{CooperativeTask} whose iterator has been exhausted
        should raise L{TaskDone}.
        """
        self.stopNext()
        self.scheduler.pump()
        self.assertRaises(task.TaskDone, self.task.stop)

    def test_stopErrored(self):
        """
        C{stop()}ping a L{CooperativeTask} whose iterator has encountered an
        error should raise L{TaskFailed}.
        """
        self.dieNext()
        self.scheduler.pump()
        self.assertRaises(task.TaskFailed, self.task.stop)

    def test_stopCooperatorReentrancy(self):
        """
        If a callback of a L{Deferred} from L{CooperativeTask.whenDone} calls
        C{Cooperator.stop} on its L{CooperativeTask._cooperator}, the
        L{Cooperator} will stop, but the L{CooperativeTask} whose callback is
        calling C{stop} should already be considered 'stopped' by the time the
        callback is running, and therefore removed from the
        L{CoooperativeTask}.
        """
        callbackPhases = []

        def stopit(result):
            callbackPhases.append(result)
            self.cooperator.stop()
            callbackPhases.append('done')
        self.task.whenDone().addCallback(stopit)
        self.stopNext()
        self.scheduler.pump()
        self.assertEqual(callbackPhases, [self.task._iterator, 'done'])