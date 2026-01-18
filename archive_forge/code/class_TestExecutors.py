import threading
import time
from eventlet.green import threading as green_threading
import testscenarios
from testtools import testcase
import futurist
from futurist import rejection
from futurist.tests import base
class TestExecutors(testscenarios.TestWithScenarios, base.TestCase):
    scenarios = [('sync', {'executor_cls': futurist.SynchronousExecutor, 'restartable': True, 'executor_kwargs': {}}), ('green_sync', {'executor_cls': futurist.SynchronousExecutor, 'restartable': True, 'executor_kwargs': {'green': True}}), ('green', {'executor_cls': futurist.GreenThreadPoolExecutor, 'restartable': False, 'executor_kwargs': {}}), ('thread', {'executor_cls': futurist.ThreadPoolExecutor, 'restartable': False, 'executor_kwargs': {}}), ('process', {'executor_cls': futurist.ProcessPoolExecutor, 'restartable': False, 'executor_kwargs': {}})]

    def setUp(self):
        super(TestExecutors, self).setUp()
        self.executor = self.executor_cls(**self.executor_kwargs)

    def tearDown(self):
        super(TestExecutors, self).tearDown()
        self.executor.shutdown()
        self.executor = None

    def test_run_one(self):
        fut = self.executor.submit(returns_one)
        self.assertEqual(1, fut.result())
        self.assertTrue(fut.done())

    def test_blows_up(self):
        fut = self.executor.submit(blows_up)
        self.assertRaises(RuntimeError, fut.result)
        self.assertIsInstance(fut.exception(), RuntimeError)

    def test_gather_stats(self):
        self.executor.submit(blows_up)
        self.executor.submit(delayed, 0.2)
        self.executor.submit(returns_one)
        self.executor.shutdown()
        self.assertEqual(3, self.executor.statistics.executed)
        self.assertEqual(1, self.executor.statistics.failures)
        self.assertGreaterEqual(self.executor.statistics.runtime, 0.199)

    def test_post_shutdown_raises(self):
        executor = self.executor_cls(**self.executor_kwargs)
        executor.shutdown()
        self.assertRaises(RuntimeError, executor.submit, returns_one)

    def test_restartable(self):
        if not self.restartable:
            raise testcase.TestSkipped('not restartable')
        else:
            executor = self.executor_cls(**self.executor_kwargs)
            fut = executor.submit(returns_one)
            self.assertEqual(1, fut.result())
            executor.shutdown()
            self.assertEqual(1, executor.statistics.executed)
            self.assertRaises(RuntimeError, executor.submit, returns_one)
            executor.restart()
            self.assertEqual(0, executor.statistics.executed)
            fut = executor.submit(returns_one)
            self.assertEqual(1, fut.result())
            self.assertEqual(1, executor.statistics.executed)
            executor.shutdown()

    def test_alive(self):
        with self.executor_cls(**self.executor_kwargs) as executor:
            self.assertTrue(executor.alive)
        self.assertFalse(executor.alive)

    def test_done_callback(self):
        happy_completed = []
        unhappy_completed = []

        def on_done(fut):
            if fut.exception():
                unhappy_completed.append(fut)
            else:
                happy_completed.append(fut)
        for i in range(0, 10):
            if i % 2 == 0:
                fut = self.executor.submit(returns_one)
            else:
                fut = self.executor.submit(blows_up)
            fut.add_done_callback(on_done)
        self.executor.shutdown()
        self.assertEqual(10, len(happy_completed) + len(unhappy_completed))
        self.assertEqual(5, len(unhappy_completed))
        self.assertEqual(5, len(happy_completed))