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
def _terminate_with_signal(self, sig):
    self._spawn()
    os.kill(self.pid, sig)
    cond = lambda: not self._get_workers()
    timeout = 5
    self._wait(cond, timeout)
    workers = self._get_workers()
    LOG.info('workers: %r' % workers)
    self.assertFalse(workers, 'No OS processes left.')