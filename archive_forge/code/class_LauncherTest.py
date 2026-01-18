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
class LauncherTest(base.ServiceBaseTestCase):

    def test_graceful_shutdown(self):
        svc = _Service()
        launcher = service.launch(self.conf, svc)
        svc.init.wait()
        launcher.stop()
        self.assertTrue(svc.cleaned_up)
        launcher.stop()

    @mock.patch('oslo_service.service.ServiceLauncher.launch_service')
    def _test_launch_single(self, workers, mock_launch):
        svc = service.Service()
        service.launch(self.conf, svc, workers=workers)
        mock_launch.assert_called_with(svc, workers=workers)

    def test_launch_none(self):
        self._test_launch_single(None)

    def test_launch_one_worker(self):
        self._test_launch_single(1)

    def test_launch_invalid_workers_number(self):
        svc = service.Service()
        for num_workers in [0, -1]:
            self.assertRaises(ValueError, service.launch, self.conf, svc, num_workers)
        for num_workers in ['0', 'a', '1']:
            self.assertRaises(TypeError, service.launch, self.conf, svc, num_workers)

    @mock.patch('signal.alarm')
    @mock.patch('oslo_service.service.ProcessLauncher.launch_service')
    def test_multiple_worker(self, mock_launch, alarm_mock):
        svc = service.Service()
        service.launch(self.conf, svc, workers=3)
        mock_launch.assert_called_with(svc, workers=3)

    def test_launch_wrong_service_base_class(self):
        svc = mock.Mock()
        self.assertRaises(TypeError, service.launch, self.conf, svc)

    @mock.patch('signal.alarm')
    @mock.patch('oslo_service.service.Services.add')
    @mock.patch('oslo_service.eventlet_backdoor.initialize_if_enabled')
    def test_check_service_base(self, initialize_if_enabled_mock, services_mock, alarm_mock):
        initialize_if_enabled_mock.return_value = None
        launcher = service.Launcher(self.conf)
        serv = _Service()
        launcher.launch_service(serv)

    @mock.patch('signal.alarm')
    @mock.patch('oslo_service.service.Services.add')
    @mock.patch('oslo_service.eventlet_backdoor.initialize_if_enabled')
    def test_check_service_base_fails(self, initialize_if_enabled_mock, services_mock, alarm_mock):
        initialize_if_enabled_mock.return_value = None
        launcher = service.Launcher(self.conf)

        class FooService(object):

            def __init__(self):
                pass
        serv = FooService()
        self.assertRaises(TypeError, launcher.launch_service, serv)