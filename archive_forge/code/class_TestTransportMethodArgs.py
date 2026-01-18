import fixtures
from unittest import mock
from oslo_config import cfg
from stevedore import driver
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
class TestTransportMethodArgs(test_utils.BaseTestCase):
    _target = oslo_messaging.Target(topic='topic', server='server')

    def test_send_defaults(self):
        t = transport.Transport(_FakeDriver(cfg.CONF))
        t._driver.send = mock.Mock()
        t._send(self._target, 'ctxt', 'message')
        t._driver.send.assert_called_once_with(self._target, 'ctxt', 'message', wait_for_reply=None, timeout=None, call_monitor_timeout=None, retry=None, transport_options=None)

    def test_send_all_args(self):
        t = transport.Transport(_FakeDriver(cfg.CONF))
        t._driver.send = mock.Mock()
        t._send(self._target, 'ctxt', 'message', wait_for_reply='wait_for_reply', timeout='timeout', call_monitor_timeout='cm_timeout', retry='retry')
        t._driver.send.assert_called_once_with(self._target, 'ctxt', 'message', wait_for_reply='wait_for_reply', timeout='timeout', call_monitor_timeout='cm_timeout', retry='retry', transport_options=None)

    def test_send_notification(self):
        t = transport.Transport(_FakeDriver(cfg.CONF))
        t._driver.send_notification = mock.Mock()
        t._send_notification(self._target, 'ctxt', 'message', version=1.0)
        t._driver.send_notification.assert_called_once_with(self._target, 'ctxt', 'message', 1.0, retry=None)

    def test_send_notification_all_args(self):
        t = transport.Transport(_FakeDriver(cfg.CONF))
        t._driver.send_notification = mock.Mock()
        t._send_notification(self._target, 'ctxt', 'message', version=1.0, retry=5)
        t._driver.send_notification.assert_called_once_with(self._target, 'ctxt', 'message', 1.0, retry=5)

    def test_listen(self):
        t = transport.Transport(_FakeDriver(cfg.CONF))
        t._driver.listen = mock.Mock()
        t._listen(self._target, 1, None)
        t._driver.listen.assert_called_once_with(self._target, 1, None)