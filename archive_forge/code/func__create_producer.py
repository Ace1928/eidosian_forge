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
def _create_producer(target):
    connection = kombu.connection.BrokerConnection(transport='memory')
    connection.transport.polling_interval = 0.0
    connection.connect()
    channel = connection.channel()
    channel._new_queue('ae.undeliver')
    if target.fanout:
        exchange = kombu.entity.Exchange(name=target.topic + '_fanout', type='fanout', durable=False, auto_delete=True)
        producer = kombu.messaging.Producer(exchange=exchange, channel=channel, routing_key=target.topic)
    elif target.server:
        exchange = kombu.entity.Exchange(name='openstack', type='topic', durable=False, auto_delete=False)
        topic = '%s.%s' % (target.topic, target.server)
        producer = kombu.messaging.Producer(exchange=exchange, channel=channel, routing_key=topic)
    else:
        exchange = kombu.entity.Exchange(name='openstack', type='topic', durable=False, auto_delete=False)
        producer = kombu.messaging.Producer(exchange=exchange, channel=channel, routing_key=target.topic)
    return (connection, producer)