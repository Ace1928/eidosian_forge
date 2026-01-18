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
class ServiceLauncherTest(ServiceTestBase):
    """Originally from nova/tests/integrated/test_multiprocess_api.py."""

    def _spawn(self):
        self.pid = self._spawn_service(workers=2)
        cond = lambda: self.workers == len(self._get_workers())
        timeout = 10
        self._wait(cond, timeout)
        workers = self._get_workers()
        self.assertEqual(len(workers), self.workers)
        return workers

    def _get_workers(self):
        f = os.popen('ps ax -o pid,ppid,command')
        f.readline()
        processes = [tuple((int(p) for p in line.strip().split()[:2])) for line in f]
        return [p for p, pp in processes if pp == self.pid]

    def test_killed_worker_recover(self):
        start_workers = self._spawn()
        LOG.info('pid of first child is %s' % start_workers[0])
        os.kill(start_workers[0], signal.SIGTERM)
        cond = lambda: start_workers != self._get_workers()
        timeout = 5
        self._wait(cond, timeout)
        end_workers = self._get_workers()
        LOG.info('workers: %r' % end_workers)
        self.assertNotEqual(start_workers, end_workers)

    def _terminate_with_signal(self, sig):
        self._spawn()
        os.kill(self.pid, sig)
        cond = lambda: not self._get_workers()
        timeout = 5
        self._wait(cond, timeout)
        workers = self._get_workers()
        LOG.info('workers: %r' % workers)
        self.assertFalse(workers, 'No OS processes left.')

    def test_terminate_sigkill(self):
        self._terminate_with_signal(signal.SIGKILL)
        status = self._reap_test()
        self.assertTrue(os.WIFSIGNALED(status))
        self.assertEqual(signal.SIGKILL, os.WTERMSIG(status))

    def test_terminate_sigterm(self):
        self._terminate_with_signal(signal.SIGTERM)
        status = self._reap_test()
        self.assertTrue(os.WIFEXITED(status))
        self.assertEqual(0, os.WEXITSTATUS(status))

    def test_crashed_service(self):
        service_maker = lambda: ServiceCrashOnStart()
        self.pid = self._spawn_service(service_maker=service_maker)
        status = self._reap_test()
        self.assertTrue(os.WIFEXITED(status))
        self.assertEqual(1, os.WEXITSTATUS(status))

    def test_child_signal_sighup(self):
        start_workers = self._spawn()
        os.kill(start_workers[0], signal.SIGHUP)
        cond = lambda: start_workers != self._get_workers()
        timeout = 5
        self._wait(cond, timeout)
        end_workers = self._get_workers()
        LOG.info('workers: %r' % end_workers)
        self.assertEqual(start_workers, end_workers)

    def test_parent_signal_sighup(self):
        start_workers = self._spawn()
        os.kill(self.pid, signal.SIGHUP)

        def cond():
            workers = self._get_workers()
            return len(workers) == len(start_workers) and (not set(start_workers).intersection(workers))
        timeout = 10
        self._wait(cond, timeout)
        self.assertTrue(cond())