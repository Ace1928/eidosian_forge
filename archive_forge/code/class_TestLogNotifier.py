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
class TestLogNotifier(test_utils.BaseTestCase):

    @mock.patch('oslo_utils.timeutils.utcnow')
    def test_notifier(self, mock_utcnow):
        self.config(driver=['log'], group='oslo_messaging_notifications')
        transport = oslo_messaging.get_notification_transport(self.conf, url='fake:')
        notifier = oslo_messaging.Notifier(transport, 'test.localhost')
        message_id = uuid.uuid4()
        uuid.uuid4 = mock.Mock()
        uuid.uuid4.return_value = message_id
        mock_utcnow.return_value = datetime.datetime.utcnow()
        logger = mock.Mock()
        message = {'message_id': str(message_id), 'publisher_id': 'test.localhost', 'event_type': 'test.notify', 'priority': 'INFO', 'payload': 'bar', 'timestamp': str(timeutils.utcnow())}
        with mock.patch.object(logging, 'getLogger') as gl:
            gl.return_value = logger
            notifier.info(test_utils.TestContext(), 'test.notify', 'bar')
            uuid.uuid4.assert_has_calls([mock.call(), mock.call()])
            logging.getLogger.assert_called_once_with('oslo.messaging.notification.test.notify')
        logger.info.assert_called_once_with(JsonMessageMatcher(message))
        self.assertTrue(notifier.is_enabled())

    def test_sample_priority(self):
        driver = _impl_log.LogDriver(None, None, None)
        logger = mock.Mock(spec=logging.getLogger('oslo.messaging.notification.foo'))
        logger.sample = None
        msg = {'event_type': 'foo'}
        with mock.patch.object(logging, 'getLogger') as gl:
            gl.return_value = logger
            driver.notify(None, msg, 'sample', None)
            logging.getLogger.assert_called_once_with('oslo.messaging.notification.foo')

    def test_mask_passwords(self):
        driver = _impl_log.LogDriver(None, None, None)
        logger = mock.MagicMock()
        logger.info = mock.MagicMock()
        message = {'password': 'passw0rd', 'event_type': 'foo'}
        mask_str = jsonutils.dumps(strutils.mask_dict_password(message))
        with mock.patch.object(logging, 'getLogger') as gl:
            gl.return_value = logger
            driver.notify(None, message, 'info', 0)
        logger.info.assert_called_once_with(mask_str)