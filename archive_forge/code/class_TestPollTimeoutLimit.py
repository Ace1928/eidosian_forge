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
class TestPollTimeoutLimit(test_utils.BaseTestCase):

    def test_poll_timeout_limit(self):
        transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
        self.addCleanup(transport.cleanup)
        driver = transport._driver
        target = oslo_messaging.Target(topic='testtopic')
        listener = driver.listen(target, None, None)._poll_style_listener
        thread = threading.Thread(target=listener.poll)
        thread.daemon = True
        thread.start()
        time.sleep(amqpdriver.ACK_REQUEUE_EVERY_SECONDS_MAX * 2)
        try:
            self.assertEqual(amqpdriver.ACK_REQUEUE_EVERY_SECONDS_MAX, listener._current_timeout)
        finally:
            driver.send(target, {}, {'tx_id': 'test'})
            thread.join()