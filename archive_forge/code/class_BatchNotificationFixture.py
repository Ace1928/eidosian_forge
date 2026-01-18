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
class BatchNotificationFixture(NotificationFixture):

    def __init__(self, conf, url, topics, batch_size=5, batch_timeout=2):
        super(BatchNotificationFixture, self).__init__(conf, url, topics)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

    def _get_server(self, transport, targets):
        return oslo_messaging.get_batch_notification_listener(transport.transport, targets, [self], 'eventlet', batch_timeout=self.batch_timeout, batch_size=self.batch_size)

    def debug(self, messages):
        self.events.put(['debug', messages])

    def audit(self, messages):
        self.events.put(['audit', messages])

    def info(self, messages):
        self.events.put(['info', messages])

    def warn(self, messages):
        self.events.put(['warn', messages])

    def error(self, messages):
        self.events.put(['error', messages])

    def critical(self, messages):
        self.events.put(['critical', messages])

    def sample(self, messages):
        pass