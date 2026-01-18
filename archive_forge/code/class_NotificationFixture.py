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
class NotificationFixture(fixtures.Fixture):

    def __init__(self, conf, url, topics, batch=None):
        super(NotificationFixture, self).__init__()
        self.conf = conf
        self.url = url
        self.topics = topics
        self.events = queue.Queue()
        self.name = str(id(self))
        self.batch = batch

    def setUp(self):
        super(NotificationFixture, self).setUp()
        targets = [oslo_messaging.Target(topic=t) for t in self.topics]
        targets.append(oslo_messaging.Target(topic=self.name))
        transport = self.useFixture(NotificationTransportFixture(self.conf, self.url))
        self.server = self._get_server(transport, targets)
        self._ctrl = self.notifier('internal', topics=[self.name])
        self._start()
        transport.wait()

    def cleanUp(self):
        self._stop()
        super(NotificationFixture, self).cleanUp()

    def _get_server(self, transport, targets):
        return oslo_messaging.get_notification_listener(transport.transport, targets, [self], 'eventlet')

    def _start(self):
        self.thread = test_utils.ServerThreadHelper(self.server)
        self.thread.start()

    def _stop(self):
        self.thread.stop()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise Exception('Server did not shutdown properly')

    def notifier(self, publisher, topics=None):
        transport = self.useFixture(NotificationTransportFixture(self.conf, self.url))
        n = notifier.Notifier(transport.transport, publisher, driver='messaging', topics=topics or self.topics)
        transport.wait()
        return n

    def debug(self, ctxt, publisher, event_type, payload, metadata):
        self.events.put(['debug', event_type, payload, publisher])

    def audit(self, ctxt, publisher, event_type, payload, metadata):
        self.events.put(['audit', event_type, payload, publisher])

    def info(self, ctxt, publisher, event_type, payload, metadata):
        self.events.put(['info', event_type, payload, publisher])

    def warn(self, ctxt, publisher, event_type, payload, metadata):
        self.events.put(['warn', event_type, payload, publisher])

    def error(self, ctxt, publisher, event_type, payload, metadata):
        self.events.put(['error', event_type, payload, publisher])

    def critical(self, ctxt, publisher, event_type, payload, metadata):
        self.events.put(['critical', event_type, payload, publisher])

    def sample(self, ctxt, publisher, event_type, payload, metadata):
        pass

    def get_events(self, timeout=0.5):
        results = []
        try:
            while True:
                results.append(self.events.get(timeout=timeout))
        except queue.Empty:
            pass
        return results