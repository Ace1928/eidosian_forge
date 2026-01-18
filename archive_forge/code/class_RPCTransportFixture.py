import os
import queue
import time
import uuid
import fixtures
from oslo_config import cfg
import oslo_messaging
from oslo_messaging._drivers.kafka_driver import kafka_options
from oslo_messaging.notify import notifier
from oslo_messaging.tests import utils as test_utils
class RPCTransportFixture(TransportFixture):
    """Fixture defined to setup RPC transport."""

    def setUp(self):
        super(RPCTransportFixture, self).setUp()
        self.transport = oslo_messaging.get_rpc_transport(self.conf, url=self.url)