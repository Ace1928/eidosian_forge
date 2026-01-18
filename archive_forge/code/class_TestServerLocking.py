import threading
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging.rpc import dispatcher
from oslo_messaging.rpc import server as rpc_server_module
from oslo_messaging import server as server_module
from oslo_messaging.tests import utils as test_utils
class TestServerLocking(test_utils.BaseTestCase):

    def setUp(self):
        super(TestServerLocking, self).setUp(conf=cfg.ConfigOpts())

        def _logmethod(name):

            def method(self, *args, **kwargs):
                with self._lock:
                    self._calls.append(name)
            return method
        executors = []

        class FakeExecutor(object):

            def __init__(self, *args, **kwargs):
                self._lock = threading.Lock()
                self._calls = []
                executors.append(self)
            submit = _logmethod('submit')
            shutdown = _logmethod('shutdown')
        self.executors = executors

        class MessageHandlingServerImpl(oslo_messaging.MessageHandlingServer):

            def _create_listener(self):
                return mock.Mock()

            def _process_incoming(self, incoming):
                pass
        self.server = MessageHandlingServerImpl(mock.Mock(), mock.Mock())
        self.server._executor_cls = FakeExecutor

    def test_start_stop_wait(self):
        eventlet.spawn(self.server.start)
        self.server.stop()
        self.server.wait()
        self.assertEqual(1, len(self.executors))
        self.assertEqual(['shutdown'], self.executors[0]._calls)
        self.assertTrue(self.server.listener.cleanup.called)

    def test_reversed_order(self):
        eventlet.spawn(self.server.wait)
        eventlet.sleep(0)
        eventlet.spawn(self.server.stop)
        eventlet.sleep(0)
        eventlet.spawn(self.server.start)
        self.server.wait()
        self.assertEqual(1, len(self.executors))
        self.assertEqual(['shutdown'], self.executors[0]._calls)

    def test_wait_for_running_task(self):
        start_event = eventletutils.Event()
        finish_event = eventletutils.Event()
        running_event = eventletutils.Event()
        done_event = eventletutils.Event()
        _runner = [None]

        class SteppingFakeExecutor(self.server._executor_cls):

            def __init__(self, *args, **kwargs):
                _runner[0] = eventlet.getcurrent()
                running_event.set()
                start_event.wait()
                super(SteppingFakeExecutor, self).__init__(*args, **kwargs)
                done_event.set()
                finish_event.wait()
        self.server._executor_cls = SteppingFakeExecutor
        start1 = eventlet.spawn(self.server.start)
        start2 = eventlet.spawn(self.server.start)
        running_event.wait()
        runner = _runner[0]
        waiter = start2 if runner == start1 else start2
        waiter_finished = eventletutils.Event()
        waiter.link(lambda _: waiter_finished.set())
        self.assertEqual(0, len(self.executors))
        self.assertFalse(waiter_finished.is_set())
        start_event.set()
        done_event.wait()
        self.assertEqual(1, len(self.executors))
        self.assertEqual([], self.executors[0]._calls)
        self.assertFalse(waiter_finished.is_set())
        finish_event.set()
        waiter.wait()
        runner.wait()
        self.assertTrue(waiter_finished.is_set())
        self.assertEqual(1, len(self.executors))
        self.assertEqual([], self.executors[0]._calls)

    def test_start_stop_wait_stop_wait(self):
        self.server.start()
        self.server.stop()
        self.server.wait()
        self.server.stop()
        self.server.wait()
        self.assertEqual(len(self.executors), 1)
        self.assertEqual(['shutdown'], self.executors[0]._calls)
        self.assertTrue(self.server.listener.cleanup.called)

    def test_state_wrapping(self):
        complete_event = eventletutils.Event()
        complete_waiting_callback = eventletutils.Event()
        start_state = self.server._states['start']
        old_wait_for_completion = start_state.wait_for_completion
        waited = [False]

        def new_wait_for_completion(*args, **kwargs):
            if not waited[0]:
                waited[0] = True
                complete_waiting_callback.set()
                complete_event.wait()
            old_wait_for_completion(*args, **kwargs)
        start_state.wait_for_completion = new_wait_for_completion
        thread1 = eventlet.spawn(self.server.stop)
        thread1_finished = eventletutils.Event()
        thread1.link(lambda _: thread1_finished.set())
        self.server.start()
        complete_waiting_callback.wait()
        self.assertEqual(1, len(self.executors))
        self.assertEqual([], self.executors[0]._calls)
        self.assertFalse(thread1_finished.is_set())
        self.server.stop()
        self.server.wait()
        self.assertEqual(1, len(self.executors))
        self.assertEqual(['shutdown'], self.executors[0]._calls)
        self.assertFalse(thread1_finished.is_set())
        self.server.start()
        self.assertEqual(2, len(self.executors))
        self.assertEqual(['shutdown'], self.executors[0]._calls)
        self.assertEqual([], self.executors[1]._calls)
        self.assertFalse(thread1_finished.is_set())
        complete_event.set()
        thread1_finished.wait()
        self.assertEqual(2, len(self.executors))
        self.assertEqual(['shutdown'], self.executors[0]._calls)
        self.assertEqual([], self.executors[1]._calls)
        self.assertTrue(thread1_finished.is_set())

    @mock.patch.object(server_module, 'DEFAULT_LOG_AFTER', 1)
    @mock.patch.object(server_module, 'LOG')
    def test_logging(self, mock_log):
        log_event = eventletutils.Event()
        mock_log.warning.side_effect = lambda _, __: log_event.set()
        thread = eventlet.spawn(self.server.stop)
        log_event.wait()
        self.assertTrue(mock_log.warning.called)
        thread.kill()

    @mock.patch.object(server_module, 'LOG')
    def test_logging_explicit_wait(self, mock_log):
        log_event = eventletutils.Event()
        mock_log.warning.side_effect = lambda _, __: log_event.set()
        thread = eventlet.spawn(self.server.stop, log_after=1)
        log_event.wait()
        self.assertTrue(mock_log.warning.called)
        thread.kill()

    @mock.patch.object(server_module, 'LOG')
    def test_logging_with_timeout(self, mock_log):
        log_event = eventletutils.Event()
        mock_log.warning.side_effect = lambda _, __: log_event.set()
        thread = eventlet.spawn(self.server.stop, log_after=1, timeout=2)
        log_event.wait()
        self.assertTrue(mock_log.warning.called)
        thread.kill()

    def test_timeout_wait(self):
        self.assertRaises(server_module.TaskTimeout, self.server.stop, timeout=1)

    def test_timeout_running(self):
        self.server.start()
        self.server.stop()
        shutdown_called = eventletutils.Event()

        def slow_shutdown(wait):
            shutdown_called.set()
            eventlet.sleep(10)
        self.executors[0].shutdown = slow_shutdown
        thread = eventlet.spawn(self.server.wait)
        shutdown_called.wait()
        self.assertRaises(server_module.TaskTimeout, self.server.wait, timeout=1)
        thread.kill()

    @mock.patch.object(server_module, 'LOG')
    def test_log_after_zero(self, mock_log):
        self.assertRaises(server_module.TaskTimeout, self.server.stop, log_after=0, timeout=2)
        self.assertFalse(mock_log.warning.called)