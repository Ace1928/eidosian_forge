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
class RpcKombuHATestCase(test_utils.BaseTestCase):

    def setUp(self):
        super(RpcKombuHATestCase, self).setUp()
        transport_url = 'rabbit:/host1,host2,host3,host4,host5/'
        self.messaging_conf.transport_url = transport_url
        self.config(rabbit_retry_interval=0.01, rabbit_retry_backoff=0.01, kombu_reconnect_delay=0, heartbeat_timeout_threshold=0, group='oslo_messaging_rabbit')
        self.useFixture(fixtures.MockPatch('kombu.connection.Connection.connection'))
        self.useFixture(fixtures.MockPatch('kombu.connection.Connection.channel'))
        if hasattr(kombu.connection.Connection, '_connection_factory'):
            self.useFixture(fixtures.MockPatch('kombu.connection.Connection._connection_factory'))
        url = oslo_messaging.TransportURL.parse(self.conf, None)
        self.connection = rabbit_driver.Connection(self.conf, url, driver_common.PURPOSE_SEND)
        if hasattr(kombu.connection.Connection, 'connect'):
            self.useFixture(fixtures.MockPatch('kombu.connection.Connection.connect'))
        self.addCleanup(self.connection.close)

    def test_ensure_four_retry(self):
        mock_callback = mock.Mock(side_effect=IOError)
        self.assertRaises(oslo_messaging.MessageDeliveryFailure, self.connection.ensure, mock_callback, retry=4)
        expected = 5
        if kombu.VERSION < (5, 2, 4):
            expected = 6
        self.assertEqual(expected, mock_callback.call_count)

    def test_ensure_one_retry(self):
        mock_callback = mock.Mock(side_effect=IOError)
        self.assertRaises(oslo_messaging.MessageDeliveryFailure, self.connection.ensure, mock_callback, retry=1)
        expected = 2
        if kombu.VERSION < (5, 2, 4):
            expected = 3
        self.assertEqual(expected, mock_callback.call_count)

    def test_ensure_no_retry(self):
        mock_callback = mock.Mock(side_effect=IOError)
        self.assertRaises(oslo_messaging.MessageDeliveryFailure, self.connection.ensure, mock_callback, retry=0)
        expected = 1
        if kombu.VERSION < (5, 2, 4):
            expected = 2
        self.assertEqual(expected, mock_callback.call_count)