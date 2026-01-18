import datetime
import logging
import sys
import uuid
import fixtures
from kombu import connection
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import timeutils
from stevedore import dispatch
from stevedore import extension
import testscenarios
import yaml
import oslo_messaging
from oslo_messaging.notify import _impl_log
from oslo_messaging.notify import _impl_test
from oslo_messaging.notify import messaging
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class TestMessagingNotifier(test_utils.BaseTestCase):
    _v1 = [('v1', dict(v1=True)), ('not_v1', dict(v1=False))]
    _v2 = [('v2', dict(v2=True)), ('not_v2', dict(v2=False))]
    _publisher_id = [('ctor_pub_id', dict(ctor_pub_id='test', expected_pub_id='test')), ('prep_pub_id', dict(prep_pub_id='test.localhost', expected_pub_id='test.localhost')), ('override', dict(ctor_pub_id='test', prep_pub_id='test.localhost', expected_pub_id='test.localhost'))]
    _topics = [('no_topics', dict(topics=[])), ('single_topic', dict(topics=['notifications'])), ('multiple_topic2', dict(topics=['foo', 'bar']))]
    _priority = [('audit', dict(priority='audit')), ('debug', dict(priority='debug')), ('info', dict(priority='info')), ('warn', dict(priority='warn')), ('error', dict(priority='error')), ('sample', dict(priority='sample')), ('critical', dict(priority='critical'))]
    _payload = [('payload', dict(payload={'foo': 'bar'}))]
    _context = [('ctxt', dict(ctxt=test_utils.TestContext(user_name='bob')))]
    _retry = [('unconfigured', dict()), ('None', dict(retry=None)), ('0', dict(retry=0)), ('5', dict(retry=5))]

    @classmethod
    def generate_scenarios(cls):
        cls.scenarios = testscenarios.multiply_scenarios(cls._v1, cls._v2, cls._publisher_id, cls._topics, cls._priority, cls._payload, cls._context, cls._retry)

    def setUp(self):
        super(TestMessagingNotifier, self).setUp()
        self.logger = self.useFixture(_ReRaiseLoggedExceptionsFixture()).logger
        self.useFixture(fixtures.MockPatchObject(messaging, 'LOG', self.logger))
        self.useFixture(fixtures.MockPatchObject(msg_notifier, '_LOG', self.logger))

    @mock.patch('oslo_utils.timeutils.utcnow')
    def test_notifier(self, mock_utcnow):
        drivers = []
        if self.v1:
            drivers.append('messaging')
        if self.v2:
            drivers.append('messagingv2')
        self.config(driver=drivers, topics=self.topics, group='oslo_messaging_notifications')
        transport = oslo_messaging.get_notification_transport(self.conf, url='fake:')
        if hasattr(self, 'ctor_pub_id'):
            notifier = oslo_messaging.Notifier(transport, publisher_id=self.ctor_pub_id)
        else:
            notifier = oslo_messaging.Notifier(transport)
        prepare_kwds = {}
        if hasattr(self, 'retry'):
            prepare_kwds['retry'] = self.retry
        if hasattr(self, 'prep_pub_id'):
            prepare_kwds['publisher_id'] = self.prep_pub_id
        if prepare_kwds:
            notifier = notifier.prepare(**prepare_kwds)
        transport._send_notification = mock.Mock()
        message_id = uuid.uuid4()
        uuid.uuid4 = mock.Mock(return_value=message_id)
        mock_utcnow.return_value = datetime.datetime.utcnow()
        message = {'message_id': str(message_id), 'publisher_id': self.expected_pub_id, 'event_type': 'test.notify', 'priority': self.priority.upper(), 'payload': self.payload, 'timestamp': str(timeutils.utcnow())}
        sends = []
        if self.v1:
            sends.append(dict(version=1.0))
        if self.v2:
            sends.append(dict(version=2.0))
        calls = []
        for send_kwargs in sends:
            for topic in self.topics:
                if hasattr(self, 'retry'):
                    send_kwargs['retry'] = self.retry
                else:
                    send_kwargs['retry'] = -1
                target = oslo_messaging.Target(topic='%s.%s' % (topic, self.priority))
                calls.append(mock.call(target, self.ctxt, message, **send_kwargs))
        method = getattr(notifier, self.priority)
        method(self.ctxt, 'test.notify', self.payload)
        uuid.uuid4.assert_called_once_with()
        transport._send_notification.assert_has_calls(calls, any_order=True)
        self.assertTrue(notifier.is_enabled())