import datetime
import ssl
import sys
import threading
import time
import uuid
import fixtures
import kombu
import kombu.connection
import kombu.transport.memory
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging._drivers import amqpdriver
from oslo_messaging._drivers import common as driver_common
from oslo_messaging._drivers import impl_rabbit as rabbit_driver
from oslo_messaging.exceptions import ConfigurationError
from oslo_messaging.exceptions import MessageDeliveryFailure
from oslo_messaging.tests import utils as test_utils
from oslo_messaging.transport import DriverLoadFailure
from unittest import mock
class ConnectionLockTestCase(test_utils.BaseTestCase):

    def _thread(self, lock, sleep, heartbeat=False):

        def thread_task():
            if heartbeat:
                with lock.for_heartbeat():
                    time.sleep(sleep)
            else:
                with lock:
                    time.sleep(sleep)
        t = threading.Thread(target=thread_task)
        t.daemon = True
        t.start()
        start = time.time()

        def get_elapsed_time():
            t.join()
            return time.time() - start
        return get_elapsed_time

    def test_workers_only(self):
        lock = rabbit_driver.ConnectionLock()
        t1 = self._thread(lock, 1)
        t2 = self._thread(lock, 1)
        self.assertAlmostEqual(1, t1(), places=0)
        self.assertAlmostEqual(2, t2(), places=0)

    def test_worker_and_heartbeat(self):
        lock = rabbit_driver.ConnectionLock()
        t1 = self._thread(lock, 1)
        t2 = self._thread(lock, 1, heartbeat=True)
        self.assertAlmostEqual(1, t1(), places=0)
        self.assertAlmostEqual(2, t2(), places=0)

    def test_workers_and_heartbeat(self):
        lock = rabbit_driver.ConnectionLock()
        t1 = self._thread(lock, 1)
        t2 = self._thread(lock, 1)
        t3 = self._thread(lock, 1)
        t4 = self._thread(lock, 1, heartbeat=True)
        t5 = self._thread(lock, 1)
        self.assertAlmostEqual(1, t1(), places=0)
        self.assertAlmostEqual(2, t4(), places=0)
        self.assertAlmostEqual(3, t2(), places=0)
        self.assertAlmostEqual(4, t3(), places=0)
        self.assertAlmostEqual(5, t5(), places=0)

    def test_heartbeat(self):
        lock = rabbit_driver.ConnectionLock()
        t1 = self._thread(lock, 1, heartbeat=True)
        t2 = self._thread(lock, 1)
        self.assertAlmostEqual(1, t1(), places=0)
        self.assertAlmostEqual(2, t2(), places=0)