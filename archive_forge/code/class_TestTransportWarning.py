from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
class TestTransportWarning(test_utils.BaseTestCase):

    @mock.patch('oslo_messaging.rpc.client.LOG')
    def test_warning_when_notifier_transport(self, log):
        transport = oslo_messaging.get_notification_transport(self.conf)
        oslo_messaging.get_rpc_client(transport, oslo_messaging.Target())
        log.warning.assert_called_once_with('Using notification transport for RPC. Please use get_rpc_transport to obtain an RPC transport instance.')