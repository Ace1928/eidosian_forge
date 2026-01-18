import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
class TaskTest(common.HeatTestCase):

    def setUp(self):
        super(TaskTest, self).setUp()
        scheduler.ENABLE_SLEEP = True
        self.mock_sleep = self.patchobject(scheduler.TaskRunner, '_sleep', return_value=None)

    def test_run(self):
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        scheduler.TaskRunner(task)()
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.assertEqual(3, self.mock_sleep.call_count)

    def test_run_as_task(self):
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        tr = scheduler.TaskRunner(task)
        rt = tr.as_task()
        for step in rt:
            pass
        self.assertTrue(tr.done())
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_not_called()

    def test_run_as_task_started(self):
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        tr = scheduler.TaskRunner(task)
        tr.start()
        for step in tr.as_task():
            pass
        self.assertTrue(tr.done())
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_not_called()

    def test_run_as_task_cancel(self):
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        tr = scheduler.TaskRunner(task)
        rt = tr.as_task()
        next(rt)
        rt.close()
        self.assertTrue(tr.done())
        task.do_step.assert_called_once_with(1)
        self.mock_sleep.assert_not_called()

    def test_run_as_task_exception(self):

        class TestException(Exception):
            pass
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        tr = scheduler.TaskRunner(task)
        rt = tr.as_task()
        next(rt)
        self.assertRaises(TestException, rt.throw, TestException)
        self.assertTrue(tr.done())
        task.do_step.assert_called_once_with(1)
        self.mock_sleep.assert_not_called()

    def test_run_as_task_swallow_exception(self):

        class TestException(Exception):
            pass

        def task():
            try:
                yield
            except TestException:
                yield
        tr = scheduler.TaskRunner(task)
        rt = tr.as_task()
        next(rt)
        rt.throw(TestException)
        self.assertFalse(tr.done())
        self.assertRaises(StopIteration, next, rt)
        self.assertTrue(tr.done())

    def test_run_delays(self):
        task = DummyTask(delays=itertools.repeat(2))
        task.do_step = mock.Mock(return_value=None)
        scheduler.TaskRunner(task)()
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_has_calls([mock.call(0), mock.call(1), mock.call(1), mock.call(1), mock.call(1), mock.call(1)])
        self.assertEqual(6, self.mock_sleep.call_count)

    def test_run_delays_dynamic(self):
        task = DummyTask(delays=[2, 4, 1])
        task.do_step = mock.Mock(return_value=None)
        scheduler.TaskRunner(task)()
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_has_calls([mock.call(0), mock.call(1), mock.call(1), mock.call(1), mock.call(1), mock.call(1), mock.call(1)])
        self.assertEqual(7, self.mock_sleep.call_count)

    def test_run_wait_time(self):
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        scheduler.TaskRunner(task)(wait_time=42)
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_has_calls([mock.call(0), mock.call(42), mock.call(42)])
        self.assertEqual(3, self.mock_sleep.call_count)

    def test_start_run(self):
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        runner.start()
        runner.run_to_completion()
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_has_calls([mock.call(1), mock.call(1)])
        self.assertEqual(2, self.mock_sleep.call_count)

    def test_start_run_wait_time(self):
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        runner.start()
        runner.run_to_completion(wait_time=24)
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_has_calls([mock.call(24), mock.call(24)])
        self.assertEqual(2, self.mock_sleep.call_count)

    def test_run_progress(self):
        progress_count = []

        def progress():
            progress_count.append(None)
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        scheduler.TaskRunner(task)(progress_callback=progress)
        self.assertEqual(task.num_steps, len(progress_count))
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_has_calls([mock.call(0), mock.call(1), mock.call(1)])
        self.assertEqual(3, self.mock_sleep.call_count)

    def test_start_run_progress(self):
        progress_count = []

        def progress():
            progress_count.append(None)
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        runner.start()
        runner.run_to_completion(progress_callback=progress)
        self.assertEqual(task.num_steps - 1, len(progress_count))
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_has_calls([mock.call(1), mock.call(1)])
        self.assertEqual(2, self.mock_sleep.call_count)

    def test_run_as_task_progress(self):
        progress_count = []

        def progress():
            progress_count.append(None)
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        tr = scheduler.TaskRunner(task)
        rt = tr.as_task(progress_callback=progress)
        for step in rt:
            pass
        self.assertEqual(task.num_steps, len(progress_count))
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_not_called()

    def test_run_progress_exception(self):

        class TestException(Exception):
            pass
        progress_count = []

        def progress():
            if progress_count:
                raise TestException
            progress_count.append(None)
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        self.assertRaises(TestException, scheduler.TaskRunner(task), progress_callback=progress)
        self.assertEqual(1, len(progress_count))
        task.do_step.assert_has_calls([mock.call(1), mock.call(2)])
        self.assertEqual(2, task.do_step.call_count)
        self.mock_sleep.assert_has_calls([mock.call(0), mock.call(1)])
        self.assertEqual(2, self.mock_sleep.call_count)

    def test_start_run_progress_exception(self):

        class TestException(Exception):
            pass
        progress_count = []

        def progress():
            if progress_count:
                raise TestException
            progress_count.append(None)
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        runner.start()
        self.assertRaises(TestException, runner.run_to_completion, progress_callback=progress)
        self.assertEqual(1, len(progress_count))
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_has_calls([mock.call(1), mock.call(1)])
        self.assertEqual(2, self.mock_sleep.call_count)

    def test_run_as_task_progress_exception(self):

        class TestException(Exception):
            pass
        progress_count = []

        def progress():
            if progress_count:
                raise TestException
            progress_count.append(None)
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        tr = scheduler.TaskRunner(task)
        rt = tr.as_task(progress_callback=progress)
        next(rt)
        next(rt)
        self.assertRaises(TestException, next, rt)
        self.assertEqual(1, len(progress_count))
        task.do_step.assert_has_calls([mock.call(1), mock.call(2)])
        self.assertEqual(2, task.do_step.call_count)
        self.mock_sleep.assert_not_called()

    def test_run_progress_exception_swallow(self):

        class TestException(Exception):
            pass
        progress_count = []

        def progress():
            try:
                if not progress_count:
                    raise TestException
            finally:
                progress_count.append(None)

        def task():
            try:
                yield
            except TestException:
                yield
        scheduler.TaskRunner(task)(progress_callback=progress)
        self.assertEqual(2, len(progress_count))
        self.mock_sleep.assert_has_calls([mock.call(0), mock.call(1)])
        self.assertEqual(2, self.mock_sleep.call_count)

    def test_start_run_progress_exception_swallow(self):

        class TestException(Exception):
            pass
        progress_count = []

        def progress():
            try:
                if not progress_count:
                    raise TestException
            finally:
                progress_count.append(None)

        def task():
            yield
            try:
                yield
            except TestException:
                yield
        runner = scheduler.TaskRunner(task)
        runner.start()
        runner.run_to_completion(progress_callback=progress)
        self.assertEqual(2, len(progress_count))
        self.mock_sleep.assert_has_calls([mock.call(1), mock.call(1)])
        self.assertEqual(2, self.mock_sleep.call_count)

    def test_run_as_task_progress_exception_swallow(self):

        class TestException(Exception):
            pass
        progress_count = []

        def progress():
            try:
                if not progress_count:
                    raise TestException
            finally:
                progress_count.append(None)

        def task():
            try:
                yield
            except TestException:
                yield
        tr = scheduler.TaskRunner(task)
        rt = tr.as_task(progress_callback=progress)
        next(rt)
        next(rt)
        self.assertRaises(StopIteration, next, rt)
        self.assertEqual(2, len(progress_count))

    def test_args(self):
        args = ['foo', 'bar']
        kwargs = {'baz': 'quux', 'blarg': 'wibble'}
        task = mock.Mock()
        runner = scheduler.TaskRunner(task, *args, **kwargs)
        runner(wait_time=None)
        task.assert_called_with(*args, **kwargs)

    def test_non_callable(self):
        self.assertRaises(AssertionError, scheduler.TaskRunner, object())

    def test_stepping(self):
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        runner.start()
        self.assertFalse(runner.step())
        self.assertTrue(runner)
        self.assertFalse(runner.step())
        self.assertTrue(runner.step())
        self.assertFalse(runner)
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3)])
        self.assertEqual(3, task.do_step.call_count)
        self.mock_sleep.assert_not_called()

    def test_start_no_steps(self):
        task = DummyTask(0)
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        runner.start()
        self.assertTrue(runner.done())
        self.assertTrue(runner.step())
        task.do_step.assert_not_called()
        self.mock_sleep.assert_not_called()

    def test_start_only(self):
        task = DummyTask()
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        self.assertFalse(runner.started())
        runner.start()
        self.assertTrue(runner.started())
        task.do_step.assert_called_once_with(1)
        self.mock_sleep.assert_not_called()

    def test_double_start(self):
        runner = scheduler.TaskRunner(DummyTask())
        runner.start()
        self.assertRaises(AssertionError, runner.start)

    def test_start_cancelled(self):
        runner = scheduler.TaskRunner(DummyTask())
        runner.cancel()
        self.assertRaises(AssertionError, runner.start)

    def test_call_double_start(self):
        runner = scheduler.TaskRunner(DummyTask())
        runner(wait_time=None)
        self.assertRaises(AssertionError, runner.start)

    def test_start_function(self):

        def task():
            pass
        runner = scheduler.TaskRunner(task)
        runner.start()
        self.assertTrue(runner.started())
        self.assertTrue(runner.done())
        self.assertTrue(runner.step())

    def test_repeated_done(self):
        task = DummyTask(0)
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        runner.start()
        self.assertTrue(runner.step())
        self.assertTrue(runner.step())
        task.do_step.assert_not_called()
        self.mock_sleep.assert_not_called()

    def test_timeout(self):
        st = timeutils.wallclock()

        def task():
            while True:
                yield
        self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.5, st + 1.5])
        runner = scheduler.TaskRunner(task)
        runner.start(timeout=1)
        self.assertTrue(runner)
        self.assertRaises(scheduler.Timeout, runner.step)
        self.assertEqual(3, timeutils.wallclock.call_count)

    def test_timeout_return(self):
        st = timeutils.wallclock()

        def task():
            while True:
                try:
                    yield
                except scheduler.Timeout:
                    return
        self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.5, st + 1.5])
        runner = scheduler.TaskRunner(task)
        runner.start(timeout=1)
        self.assertTrue(runner)
        self.assertTrue(runner.step())
        self.assertFalse(runner)
        self.assertEqual(3, timeutils.wallclock.call_count)

    def test_timeout_swallowed(self):
        st = timeutils.wallclock()

        def task():
            while True:
                try:
                    yield
                except scheduler.Timeout:
                    yield
                    self.fail('Task still running')
        self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.5, st + 1.5])
        runner = scheduler.TaskRunner(task)
        runner.start(timeout=1)
        self.assertTrue(runner)
        self.assertTrue(runner.step())
        self.assertFalse(runner)
        self.assertTrue(runner.step())
        self.assertEqual(3, timeutils.wallclock.call_count)

    def test_as_task_timeout(self):
        st = timeutils.wallclock()

        def task():
            while True:
                yield
        self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.5, st + 1.5])
        runner = scheduler.TaskRunner(task)
        rt = runner.as_task(timeout=1)
        next(rt)
        self.assertTrue(runner)
        self.assertRaises(scheduler.Timeout, next, rt)
        self.assertEqual(3, timeutils.wallclock.call_count)

    def test_as_task_timeout_shorter(self):
        st = timeutils.wallclock()

        def task():
            while True:
                yield
        self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.5, st + 0.7, st + 1.6, st + 2.6])
        runner = scheduler.TaskRunner(task)
        runner.start(timeout=10)
        self.assertTrue(runner)
        rt = runner.as_task(timeout=1)
        next(rt)
        self.assertRaises(scheduler.Timeout, next, rt)
        self.assertEqual(5, timeutils.wallclock.call_count)

    def test_as_task_timeout_longer(self):
        st = timeutils.wallclock()

        def task():
            while True:
                yield
        self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.5, st + 0.6, st + 1.5])
        runner = scheduler.TaskRunner(task)
        runner.start(timeout=1)
        self.assertTrue(runner)
        rt = runner.as_task(timeout=10)
        self.assertRaises(scheduler.Timeout, next, rt)
        self.assertEqual(4, timeutils.wallclock.call_count)

    def test_cancel_not_started(self):
        task = DummyTask(1)
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        self.assertFalse(runner.started())
        runner.cancel()
        self.assertTrue(runner.done())
        task.do_step.assert_not_called()
        self.mock_sleep.assert_not_called()

    def test_cancel_done(self):
        task = DummyTask(1)
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        self.assertFalse(runner.started())
        runner.start()
        self.assertTrue(runner.started())
        self.assertTrue(runner.step())
        self.assertTrue(runner.done())
        runner.cancel()
        self.assertTrue(runner.done())
        self.assertTrue(runner.step())
        task.do_step.assert_called_once_with(1)
        self.mock_sleep.assert_not_called()

    def test_cancel(self):
        task = DummyTask(3)
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        self.assertFalse(runner.started())
        runner.start()
        self.assertTrue(runner.started())
        self.assertFalse(runner.step())
        runner.cancel()
        self.assertTrue(runner.step())
        task.do_step.assert_has_calls([mock.call(1), mock.call(2)])
        self.assertEqual(2, task.do_step.call_count)
        self.mock_sleep.assert_not_called()

    def test_cancel_grace_period(self):
        st = timeutils.wallclock()
        task = DummyTask(5)
        task.do_step = mock.Mock(return_value=None)
        self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.5, st + 1.0, st + 1.5])
        runner = scheduler.TaskRunner(task)
        self.assertFalse(runner.started())
        runner.start()
        self.assertTrue(runner.started())
        self.assertFalse(runner.step())
        runner.cancel(grace_period=1.0)
        self.assertFalse(runner.step())
        self.assertFalse(runner.step())
        self.assertTrue(runner.step())
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3), mock.call(4)])
        self.assertEqual(4, task.do_step.call_count)
        self.mock_sleep.assert_not_called()
        self.assertEqual(4, timeutils.wallclock.call_count)

    def test_cancel_grace_period_before_timeout(self):
        st = timeutils.wallclock()
        task = DummyTask(5)
        task.do_step = mock.Mock(return_value=None)
        self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.1, st + 0.2, st + 0.2, st + 0.5, st + 1.0, st + 1.5])
        runner = scheduler.TaskRunner(task)
        self.assertFalse(runner.started())
        runner.start(timeout=10)
        self.assertTrue(runner.started())
        self.assertFalse(runner.step())
        runner.cancel(grace_period=1.0)
        self.assertFalse(runner.step())
        self.assertFalse(runner.step())
        self.assertTrue(runner.step())
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3), mock.call(4)])
        self.assertEqual(4, task.do_step.call_count)
        self.mock_sleep.assert_not_called()
        self.assertEqual(7, timeutils.wallclock.call_count)

    def test_cancel_grace_period_after_timeout(self):
        st = timeutils.wallclock()
        task = DummyTask(5)
        task.do_step = mock.Mock(return_value=None)
        self.patchobject(timeutils, 'wallclock', side_effect=[st, st + 0.1, st + 0.2, st + 0.2, st + 0.5, st + 1.0, st + 1.5])
        runner = scheduler.TaskRunner(task)
        self.assertFalse(runner.started())
        runner.start(timeout=1.25)
        self.assertTrue(runner.started())
        self.assertFalse(runner.step())
        runner.cancel(grace_period=3)
        self.assertFalse(runner.step())
        self.assertFalse(runner.step())
        self.assertRaises(scheduler.Timeout, runner.step)
        task.do_step.assert_has_calls([mock.call(1), mock.call(2), mock.call(3), mock.call(4)])
        self.assertEqual(4, task.do_step.call_count)
        self.mock_sleep.assert_not_called()
        self.assertEqual(7, timeutils.wallclock.call_count)

    def test_cancel_grace_period_not_started(self):
        task = DummyTask(1)
        task.do_step = mock.Mock(return_value=None)
        runner = scheduler.TaskRunner(task)
        self.assertFalse(runner.started())
        runner.cancel(grace_period=0.5)
        self.assertTrue(runner.done())
        task.do_step.assert_not_called()
        self.mock_sleep.assert_not_called()