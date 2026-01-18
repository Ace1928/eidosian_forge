import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
class ThreadPoolTests(unittest.SynchronousTestCase):
    """
    Test threadpools.
    """

    def getTimeout(self):
        """
        Return number of seconds to wait before giving up.
        """
        return 5

    def _waitForLock(self, lock):
        items = range(1000000)
        for i in items:
            if lock.acquire(False):
                break
            time.sleep(1e-05)
        else:
            self.fail('A long time passed without succeeding')

    def test_attributes(self):
        """
        L{ThreadPool.min} and L{ThreadPool.max} are set to the values passed to
        L{ThreadPool.__init__}.
        """
        pool = threadpool.ThreadPool(12, 22)
        self.assertEqual(pool.min, 12)
        self.assertEqual(pool.max, 22)

    def test_start(self):
        """
        L{ThreadPool.start} creates the minimum number of threads specified.
        """
        pool = threadpool.ThreadPool(0, 5)
        pool.start()
        self.addCleanup(pool.stop)
        self.assertEqual(len(pool.threads), 0)
        pool = threadpool.ThreadPool(3, 10)
        self.assertEqual(len(pool.threads), 0)
        pool.start()
        self.addCleanup(pool.stop)
        self.assertEqual(len(pool.threads), 3)

    def test_adjustingWhenPoolStopped(self):
        """
        L{ThreadPool.adjustPoolsize} only modifies the pool size and does not
        start new workers while the pool is not running.
        """
        pool = threadpool.ThreadPool(0, 5)
        pool.start()
        pool.stop()
        pool.adjustPoolsize(2)
        self.assertEqual(len(pool.threads), 0)

    def test_threadCreationArguments(self):
        """
        Test that creating threads in the threadpool with application-level
        objects as arguments doesn't results in those objects never being
        freed, with the thread maintaining a reference to them as long as it
        exists.
        """
        tp = threadpool.ThreadPool(0, 1)
        tp.start()
        self.addCleanup(tp.stop)
        self.assertEqual(tp.threads, [])

        def worker(arg):
            pass

        class Dumb:
            pass
        unique = Dumb()
        workerRef = weakref.ref(worker)
        uniqueRef = weakref.ref(unique)
        tp.callInThread(worker, unique)
        event = threading.Event()
        tp.callInThread(event.set)
        event.wait(self.getTimeout())
        del worker
        del unique
        gc.collect()
        self.assertIsNone(uniqueRef())
        self.assertIsNone(workerRef())

    def test_threadCreationArgumentsCallInThreadWithCallback(self):
        """
        As C{test_threadCreationArguments} above, but for
        callInThreadWithCallback.
        """
        tp = threadpool.ThreadPool(0, 1)
        tp.start()
        self.addCleanup(tp.stop)
        self.assertEqual(tp.threads, [])
        refdict = {}
        onResultWait = threading.Event()
        onResultDone = threading.Event()
        resultRef = []

        def onResult(success, result):
            gc.collect()
            onResultWait.wait(self.getTimeout())
            refdict['workerRef'] = workerRef()
            refdict['uniqueRef'] = uniqueRef()
            onResultDone.set()
            resultRef.append(weakref.ref(result))

        def worker(arg, test):
            return Dumb()

        class Dumb:
            pass
        unique = Dumb()
        onResultRef = weakref.ref(onResult)
        workerRef = weakref.ref(worker)
        uniqueRef = weakref.ref(unique)
        tp.callInThreadWithCallback(onResult, worker, unique, test=unique)
        del worker
        del unique
        onResultWait.set()
        onResultDone.wait(self.getTimeout())
        gc.collect()
        self.assertIsNone(uniqueRef())
        self.assertIsNone(workerRef())
        del onResult
        gc.collect()
        self.assertIsNone(onResultRef())
        self.assertIsNone(resultRef[0]())
        self.assertEqual(list(refdict.values()), [None, None])

    def test_persistence(self):
        """
        Threadpools can be pickled and unpickled, which should preserve the
        number of threads and other parameters.
        """
        pool = threadpool.ThreadPool(7, 20)
        self.assertEqual(pool.min, 7)
        self.assertEqual(pool.max, 20)
        copy = pickle.loads(pickle.dumps(pool))
        self.assertEqual(copy.min, 7)
        self.assertEqual(copy.max, 20)

    def _threadpoolTest(self, method):
        """
        Test synchronization of calls made with C{method}, which should be
        one of the mechanisms of the threadpool to execute work in threads.
        """
        N = 10
        tp = threadpool.ThreadPool()
        tp.start()
        self.addCleanup(tp.stop)
        waiting = threading.Lock()
        waiting.acquire()
        actor = Synchronization(N, waiting)
        for i in range(N):
            method(tp, actor)
        self._waitForLock(waiting)
        self.assertFalse(actor.failures, f'run() re-entered {actor.failures} times')

    def test_callInThread(self):
        """
        Call C{_threadpoolTest} with C{callInThread}.
        """
        return self._threadpoolTest(lambda tp, actor: tp.callInThread(actor.run))

    def test_callInThreadException(self):
        """
        L{ThreadPool.callInThread} logs exceptions raised by the callable it
        is passed.
        """

        class NewError(Exception):
            pass

        def raiseError():
            raise NewError()
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThread(raiseError)
        tp.start()
        tp.stop()
        errors = self.flushLoggedErrors(NewError)
        self.assertEqual(len(errors), 1)

    def test_callInThreadWithCallback(self):
        """
        L{ThreadPool.callInThreadWithCallback} calls C{onResult} with a
        two-tuple of C{(True, result)} where C{result} is the value returned
        by the callable supplied.
        """
        waiter = threading.Lock()
        waiter.acquire()
        results = []

        def onResult(success, result):
            waiter.release()
            results.append(success)
            results.append(result)
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThreadWithCallback(onResult, lambda: 'test')
        tp.start()
        try:
            self._waitForLock(waiter)
        finally:
            tp.stop()
        self.assertTrue(results[0])
        self.assertEqual(results[1], 'test')

    def test_callInThreadWithCallbackExceptionInCallback(self):
        """
        L{ThreadPool.callInThreadWithCallback} calls C{onResult} with a
        two-tuple of C{(False, failure)} where C{failure} represents the
        exception raised by the callable supplied.
        """

        class NewError(Exception):
            pass

        def raiseError():
            raise NewError()
        waiter = threading.Lock()
        waiter.acquire()
        results = []

        def onResult(success, result):
            waiter.release()
            results.append(success)
            results.append(result)
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThreadWithCallback(onResult, raiseError)
        tp.start()
        try:
            self._waitForLock(waiter)
        finally:
            tp.stop()
        self.assertFalse(results[0])
        self.assertIsInstance(results[1], failure.Failure)
        self.assertTrue(issubclass(results[1].type, NewError))

    def test_callInThreadWithCallbackExceptionInOnResult(self):
        """
        L{ThreadPool.callInThreadWithCallback} logs the exception raised by
        C{onResult}.
        """

        class NewError(Exception):
            pass
        waiter = threading.Lock()
        waiter.acquire()
        results = []

        def onResult(success, result):
            results.append(success)
            results.append(result)
            raise NewError()
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThreadWithCallback(onResult, lambda: None)
        tp.callInThread(waiter.release)
        tp.start()
        try:
            self._waitForLock(waiter)
        finally:
            tp.stop()
        errors = self.flushLoggedErrors(NewError)
        self.assertEqual(len(errors), 1)
        self.assertTrue(results[0])
        self.assertIsNone(results[1])

    def test_callbackThread(self):
        """
        L{ThreadPool.callInThreadWithCallback} calls the function it is
        given and the C{onResult} callback in the same thread.
        """
        threadIds = []
        event = threading.Event()

        def onResult(success, result):
            threadIds.append(threading.current_thread().ident)
            event.set()

        def func():
            threadIds.append(threading.current_thread().ident)
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThreadWithCallback(onResult, func)
        tp.start()
        self.addCleanup(tp.stop)
        event.wait(self.getTimeout())
        self.assertEqual(len(threadIds), 2)
        self.assertEqual(threadIds[0], threadIds[1])

    def test_callbackContext(self):
        """
        The context L{ThreadPool.callInThreadWithCallback} is invoked in is
        shared by the context the callable and C{onResult} callback are
        invoked in.
        """
        myctx = context.theContextTracker.currentContext().contexts[-1]
        myctx['testing'] = 'this must be present'
        contexts = []
        event = threading.Event()

        def onResult(success, result):
            ctx = context.theContextTracker.currentContext().contexts[-1]
            contexts.append(ctx)
            event.set()

        def func():
            ctx = context.theContextTracker.currentContext().contexts[-1]
            contexts.append(ctx)
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThreadWithCallback(onResult, func)
        tp.start()
        self.addCleanup(tp.stop)
        event.wait(self.getTimeout())
        self.assertEqual(len(contexts), 2)
        self.assertEqual(myctx, contexts[0])
        self.assertEqual(myctx, contexts[1])

    def test_existingWork(self):
        """
        Work added to the threadpool before its start should be executed once
        the threadpool is started: this is ensured by trying to release a lock
        previously acquired.
        """
        waiter = threading.Lock()
        waiter.acquire()
        tp = threadpool.ThreadPool(0, 1)
        tp.callInThread(waiter.release)
        tp.start()
        try:
            self._waitForLock(waiter)
        finally:
            tp.stop()

    def test_workerStateTransition(self):
        """
        As the worker receives and completes work, it transitions between
        the working and waiting states.
        """
        pool = threadpool.ThreadPool(0, 1)
        pool.start()
        self.addCleanup(pool.stop)
        self.assertEqual(pool.workers, 0)
        self.assertEqual(len(pool.waiters), 0)
        self.assertEqual(len(pool.working), 0)
        threadWorking = threading.Event()
        threadFinish = threading.Event()

        def _thread():
            threadWorking.set()
            threadFinish.wait(10)
        pool.callInThread(_thread)
        threadWorking.wait(10)
        self.assertEqual(pool.workers, 1)
        self.assertEqual(len(pool.waiters), 0)
        self.assertEqual(len(pool.working), 1)
        threadFinish.set()
        while not len(pool.waiters):
            time.sleep(0.0005)
        self.assertEqual(len(pool.waiters), 1)
        self.assertEqual(len(pool.working), 0)

    def test_q(self) -> None:
        """
        There is a property '_queue' for legacy purposes
        """
        pool = threadpool.ThreadPool(0, 1)
        self.assertEqual(pool._queue.qsize(), 0)