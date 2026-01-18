import logging
import multiprocessing
import os
import signal
import socket
import time
import traceback
from unittest import mock
import eventlet
from eventlet import event
from oslotest import base as test_base
from oslo_service import service
from oslo_service.tests import base
from oslo_service.tests import eventlet_service
class ProcessLauncherTest(base.ServiceBaseTestCase):

    @mock.patch('signal.alarm')
    @mock.patch('signal.signal')
    def test_stop(self, signal_mock, alarm_mock):
        signal_mock.SIGTERM = 15
        launcher = service.ProcessLauncher(self.conf)
        self.assertTrue(launcher.running)
        pid_nums = [22, 222]
        fakeServiceWrapper = service.ServiceWrapper(service.Service(), 1)
        launcher.children = {pid_nums[0]: fakeServiceWrapper, pid_nums[1]: fakeServiceWrapper}
        with mock.patch('oslo_service.service.os.kill') as mock_kill:
            with mock.patch.object(launcher, '_wait_child') as _wait_child:

                def fake_wait_child():
                    pid = pid_nums.pop()
                    return launcher.children.pop(pid)
                _wait_child.side_effect = fake_wait_child
                with mock.patch('oslo_service.service.Service.stop') as mock_service_stop:
                    mock_service_stop.side_effect = lambda: None
                    launcher.stop()
        self.assertFalse(launcher.running)
        self.assertFalse(launcher.children)
        mock_kill.assert_has_calls([mock.call(222, signal_mock.SIGTERM), mock.call(22, signal_mock.SIGTERM)], any_order=True)
        self.assertEqual(2, mock_kill.call_count)
        mock_service_stop.assert_called_once_with()

    def test__handle_signal(self):
        signal_handler = service.SignalHandler()
        signal_handler.clear()
        self.assertEqual(0, len(signal_handler._signal_handlers[signal.SIGTERM]))
        call_1, call_2 = (mock.Mock(), mock.Mock())
        signal_handler.add_handler('SIGTERM', call_1)
        signal_handler.add_handler('SIGTERM', call_2)
        self.assertEqual(2, len(signal_handler._signal_handlers[signal.SIGTERM]))
        signal_handler._handle_signal(signal.SIGTERM, 'test')
        time.sleep(0)
        for m in signal_handler._signal_handlers[signal.SIGTERM]:
            m.assert_called_once_with(signal.SIGTERM, 'test')
        signal_handler.clear()

    def test_setup_signal_interruption_no_select_poll(self):
        service.SignalHandler.__class__._instances.clear()
        with mock.patch('eventlet.patcher.original', return_value=object()) as get_original:
            signal_handler = service.SignalHandler()
            get_original.assert_called_with('select')
        self.addCleanup(service.SignalHandler.__class__._instances.clear)
        self.assertFalse(signal_handler._SignalHandler__force_interrupt_on_signal)

    def test_setup_signal_interruption_select_poll(self):
        service.SignalHandler.__class__._instances.clear()
        signal_handler = service.SignalHandler()
        self.addCleanup(service.SignalHandler.__class__._instances.clear)
        self.assertTrue(signal_handler._SignalHandler__force_interrupt_on_signal)

    @mock.patch('signal.alarm')
    @mock.patch('os.kill')
    @mock.patch('oslo_service.service.ProcessLauncher.stop')
    @mock.patch('oslo_service.service.ProcessLauncher._respawn_children')
    @mock.patch('oslo_service.service.ProcessLauncher.handle_signal')
    @mock.patch('oslo_config.cfg.CONF.log_opt_values')
    @mock.patch('oslo_service.systemd.notify_once')
    @mock.patch('oslo_config.cfg.CONF.reload_config_files')
    @mock.patch('oslo_service.service._is_sighup_and_daemon')
    def test_parent_process_reload_config(self, is_sighup_and_daemon_mock, reload_config_files_mock, notify_once_mock, log_opt_values_mock, handle_signal_mock, respawn_children_mock, stop_mock, kill_mock, alarm_mock):
        is_sighup_and_daemon_mock.return_value = True
        respawn_children_mock.side_effect = [None, eventlet.greenlet.GreenletExit()]
        launcher = service.ProcessLauncher(self.conf)
        launcher.sigcaught = 1
        launcher.children = {}
        wrap_mock = mock.Mock()
        launcher.children[222] = wrap_mock
        launcher.wait()
        reload_config_files_mock.assert_called_once_with()
        wrap_mock.service.reset.assert_called_once_with()

    @mock.patch('oslo_service.service.ProcessLauncher._start_child')
    @mock.patch('oslo_service.service.ProcessLauncher.handle_signal')
    @mock.patch('eventlet.greenio.GreenPipe')
    @mock.patch('os.pipe')
    def test_check_service_base(self, pipe_mock, green_pipe_mock, handle_signal_mock, start_child_mock):
        pipe_mock.return_value = [None, None]
        launcher = service.ProcessLauncher(self.conf)
        serv = _Service()
        launcher.launch_service(serv, workers=0)

    @mock.patch('oslo_service.service.ProcessLauncher._start_child')
    @mock.patch('oslo_service.service.ProcessLauncher.handle_signal')
    @mock.patch('eventlet.greenio.GreenPipe')
    @mock.patch('os.pipe')
    def test_check_service_base_fails(self, pipe_mock, green_pipe_mock, handle_signal_mock, start_child_mock):
        pipe_mock.return_value = [None, None]
        launcher = service.ProcessLauncher(self.conf)

        class FooService(object):

            def __init__(self):
                pass
        serv = FooService()
        self.assertRaises(TypeError, launcher.launch_service, serv, 0)

    @mock.patch('oslo_service.service.ProcessLauncher._start_child')
    @mock.patch('oslo_service.service.ProcessLauncher.handle_signal')
    @mock.patch('eventlet.greenio.GreenPipe')
    @mock.patch('os.pipe')
    def test_double_sighup(self, pipe_mock, green_pipe_mock, handle_signal_mock, start_child_mock):
        pipe_mock.return_value = [None, None]
        launcher = service.ProcessLauncher(self.conf)
        serv = _Service()
        launcher.launch_service(serv, workers=0)

        def stager():
            stager.stage += 1
            if stager.stage < 3:
                launcher._handle_hup(1, mock.sentinel.frame)
            elif stager.stage == 3:
                launcher._handle_term(15, mock.sentinel.frame)
            else:
                self.fail('TERM did not kill launcher')
        stager.stage = -1
        handle_signal_mock.side_effect = stager
        launcher.wait()
        self.assertEqual(3, stager.stage)