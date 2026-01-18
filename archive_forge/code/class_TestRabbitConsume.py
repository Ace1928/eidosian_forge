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
class TestRabbitConsume(test_utils.BaseTestCase):

    def test_consume_timeout(self):
        transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
        self.addCleanup(transport.cleanup)
        deadline = time.time() + 6
        with transport._driver._get_connection(driver_common.PURPOSE_LISTEN) as conn:
            self.assertRaises(driver_common.Timeout, conn.consume, timeout=3)
            conn.connection.connection.recoverable_channel_errors = (IOError,)
            conn.declare_fanout_consumer('notif.info', lambda msg: True)
            with mock.patch('kombu.connection.Connection.drain_events', side_effect=IOError):
                self.assertRaises(driver_common.Timeout, conn.consume, timeout=3)
        self.assertEqual(0, int(deadline - time.time()))

    def test_consume_from_missing_queue(self):
        transport = oslo_messaging.get_transport(self.conf, 'kombu+memory://')
        self.addCleanup(transport.cleanup)
        with transport._driver._get_connection(driver_common.PURPOSE_LISTEN) as conn:
            with mock.patch('kombu.Queue.consume') as consume, mock.patch('kombu.Queue.declare') as declare:
                conn.declare_topic_consumer(exchange_name='test', topic='test', callback=lambda msg: True)
                import amqp
                consume.side_effect = [amqp.NotFound, None]
                conn.connection.connection.recoverable_connection_errors = ()
                conn.connection.connection.recoverable_channel_errors = ()
                self.assertEqual(1, declare.call_count)
                conn.connection.connection.drain_events = mock.Mock()
                conn.consume()
                self.assertEqual(2, declare.call_count)

    def test_consume_from_missing_queue_with_io_error_on_redeclaration(self):
        transport = oslo_messaging.get_transport(self.conf, 'kombu+memory://')
        self.addCleanup(transport.cleanup)
        with transport._driver._get_connection(driver_common.PURPOSE_LISTEN) as conn:
            with mock.patch('kombu.Queue.consume') as consume, mock.patch('kombu.Queue.declare') as declare:
                conn.declare_topic_consumer(exchange_name='test', topic='test', callback=lambda msg: True)
                import amqp
                consume.side_effect = [amqp.NotFound, None]
                declare.side_effect = [IOError, None]
                conn.connection.connection.recoverable_connection_errors = (IOError,)
                conn.connection.connection.recoverable_channel_errors = ()
                self.assertEqual(1, declare.call_count)
                conn.connection.connection.drain_events = mock.Mock()
                conn.consume()
                self.assertEqual(3, declare.call_count)

    def test_connection_ack_have_disconnected_kombu_connection(self):
        transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
        self.addCleanup(transport.cleanup)
        with transport._driver._get_connection(driver_common.PURPOSE_LISTEN) as conn:
            channel = conn.connection.channel
            with mock.patch('kombu.connection.Connection.connected', new_callable=mock.PropertyMock, return_value=False):
                self.assertRaises(driver_common.Timeout, conn.connection.consume, timeout=0.01)
                self.assertNotEqual(channel, conn.connection.channel)