import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
class SemaphoreTest(AsyncTestCase):

    def test_negative_value(self):
        self.assertRaises(ValueError, locks.Semaphore, value=-1)

    def test_repr(self):
        sem = locks.Semaphore()
        self.assertIn('Semaphore', repr(sem))
        self.assertIn('unlocked,value:1', repr(sem))
        sem.acquire()
        self.assertIn('locked', repr(sem))
        self.assertNotIn('waiters', repr(sem))
        sem.acquire()
        self.assertIn('waiters', repr(sem))

    def test_acquire(self):
        sem = locks.Semaphore()
        f0 = asyncio.ensure_future(sem.acquire())
        self.assertTrue(f0.done())
        f1 = asyncio.ensure_future(sem.acquire())
        self.assertFalse(f1.done())
        f2 = asyncio.ensure_future(sem.acquire())
        sem.release()
        self.assertTrue(f1.done())
        self.assertFalse(f2.done())
        sem.release()
        self.assertTrue(f2.done())
        sem.release()
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())
        self.assertEqual(0, len(sem._waiters))

    @gen_test
    def test_acquire_timeout(self):
        sem = locks.Semaphore(2)
        yield sem.acquire()
        yield sem.acquire()
        acquire = sem.acquire(timedelta(seconds=0.01))
        self.io_loop.call_later(0.02, sem.release)
        yield gen.sleep(0.3)
        with self.assertRaises(gen.TimeoutError):
            yield acquire
        sem.acquire()
        f = asyncio.ensure_future(sem.acquire())
        self.assertFalse(f.done())
        sem.release()
        self.assertTrue(f.done())

    @gen_test
    def test_acquire_timeout_preempted(self):
        sem = locks.Semaphore(1)
        yield sem.acquire()
        self.io_loop.call_later(0.01, sem.release)
        acquire = sem.acquire(timedelta(seconds=0.02))
        yield gen.sleep(0.03)
        yield acquire

    def test_release_unacquired(self):
        sem = locks.Semaphore()
        sem.release()
        sem.release()
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())
        self.assertTrue(asyncio.ensure_future(sem.acquire()).done())
        self.assertFalse(asyncio.ensure_future(sem.acquire()).done())

    @gen_test
    def test_garbage_collection(self):
        sem = locks.Semaphore(value=0)
        futures = [asyncio.ensure_future(sem.acquire(timedelta(seconds=0.01))) for _ in range(101)]
        future = asyncio.ensure_future(sem.acquire())
        self.assertEqual(102, len(sem._waiters))
        yield gen.sleep(0.02)
        self.assertEqual(1, len(sem._waiters))
        self.assertFalse(future.done())
        sem.release()
        self.assertTrue(future.done())
        for future in futures:
            self.assertRaises(TimeoutError, future.result)