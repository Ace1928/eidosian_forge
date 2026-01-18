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
class TestRabbitDriverLoadSSL(test_utils.BaseTestCase):
    scenarios = [('no_ssl', dict(options=dict(), expected=False)), ('no_ssl_with_options', dict(options=dict(ssl_version='TLSv1'), expected=False)), ('just_ssl', dict(options=dict(ssl=True), expected=True)), ('ssl_with_options', dict(options=dict(ssl=True, ssl_version='TLSv1', ssl_key_file='foo', ssl_cert_file='bar', ssl_ca_file='foobar'), expected=dict(ssl_version=3, keyfile='foo', certfile='bar', ca_certs='foobar', cert_reqs=ssl.CERT_REQUIRED)))]

    @mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.ensure_connection')
    @mock.patch('kombu.connection.Connection')
    def test_driver_load(self, connection_klass, fake_ensure):
        self.config(group='oslo_messaging_rabbit', **self.options)
        transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
        self.addCleanup(transport.cleanup)
        connection = transport._driver._get_connection()
        connection_klass.assert_called_once_with('memory:///', transport_options={'client_properties': {'capabilities': {'connection.blocked': True, 'consumer_cancel_notify': True, 'authentication_failure_close': True}, 'connection_name': connection.name}, 'confirm_publish': True, 'on_blocked': mock.ANY, 'on_unblocked': mock.ANY}, ssl=self.expected, login_method='AMQPLAIN', heartbeat=60, failover_strategy='round-robin')