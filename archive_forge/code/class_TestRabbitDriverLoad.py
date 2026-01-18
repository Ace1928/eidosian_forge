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
class TestRabbitDriverLoad(test_utils.BaseTestCase):
    scenarios = [('rabbit', dict(transport_url='rabbit:/guest:guest@localhost:5672//')), ('kombu', dict(transport_url='kombu:/guest:guest@localhost:5672//')), ('rabbit+memory', dict(transport_url='kombu+memory:/'))]

    @mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.ensure_connection')
    @mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.reset')
    def test_driver_load(self, fake_ensure, fake_reset):
        self.config(heartbeat_timeout_threshold=60, group='oslo_messaging_rabbit')
        self.messaging_conf.transport_url = self.transport_url
        transport = oslo_messaging.get_transport(self.conf)
        self.addCleanup(transport.cleanup)
        driver = transport._driver
        self.assertIsInstance(driver, rabbit_driver.RabbitDriver)

    @mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.ensure_connection')
    @mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.reset')
    def test_driver_load_max_less_than_min(self, fake_ensure, fake_reset):
        self.config(rpc_conn_pool_size=1, conn_pool_min_size=2, group='oslo_messaging_rabbit')
        self.messaging_conf.transport_url = self.transport_url
        error = self.assertRaises(DriverLoadFailure, oslo_messaging.get_transport, self.conf)
        self.assertIn('rpc_conn_pool_size: 1 must be greater than or equal to conn_pool_min_size: 2', str(error))