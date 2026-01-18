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
class TransportFixture(fixtures.Fixture):
    """Fixture defined to setup the oslo_messaging transport."""

    def __init__(self, conf, url):
        self.conf = conf
        self.url = url

    def setUp(self):
        super(TransportFixture, self).setUp()
        self.transport = oslo_messaging.get_transport(self.conf, url=self.url)

    def cleanUp(self):
        try:
            self.transport.cleanup()
        except fixtures.TimeoutException:
            pass
        super(TransportFixture, self).cleanUp()

    def wait(self):
        time.sleep(0.5)