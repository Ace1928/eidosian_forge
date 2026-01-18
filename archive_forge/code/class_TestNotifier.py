import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
class TestNotifier(utils.BaseTestCase):

    @mock.patch.object(oslo_messaging, 'Notifier')
    @mock.patch.object(oslo_messaging, 'get_notification_transport')
    def _test_load_strategy(self, mock_get_transport, mock_notifier, url, driver):
        nfier = notifier.Notifier()
        mock_get_transport.assert_called_with(cfg.CONF)
        self.assertIsNotNone(nfier._transport)
        mock_notifier.assert_called_with(nfier._transport, publisher_id='image.localhost')
        self.assertIsNotNone(nfier._notifier)

    def test_notifier_load(self):
        self._test_load_strategy(url=None, driver=None)

    @mock.patch.object(oslo_messaging, 'set_transport_defaults')
    def test_set_defaults(self, mock_set_trans_defaults):
        notifier.set_defaults(control_exchange='foo')
        mock_set_trans_defaults.assert_called_with('foo')
        notifier.set_defaults()
        mock_set_trans_defaults.assert_called_with('glance')