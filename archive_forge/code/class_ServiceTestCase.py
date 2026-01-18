from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
class ServiceTestCase(base.BaseTestCase):

    def setUp(self):
        super(ServiceTestCase, self).setUp()
        self.host = 'foo'
        self.topic = 'neutron-agent'
        self.target_mock = mock.patch('oslo_messaging.Target')
        self.target_mock.start()
        self.messaging_conf = messaging_conffixture.ConfFixture(CONF)
        self.messaging_conf.transport_url = 'fake://'
        self.messaging_conf.response_timeout = 0
        self.useFixture(self.messaging_conf)
        self.addCleanup(rpc.cleanup)
        rpc.init(CONF)

    @mock.patch.object(cfg, 'CONF')
    def test_operations(self, mock_conf):
        mock_conf.host = self.host
        with mock.patch('oslo_messaging.get_rpc_server') as get_rpc_server:
            rpc_server = get_rpc_server.return_value
            service = rpc.Service(self.host, self.topic)
            service.start()
            rpc_server.start.assert_called_once_with()
            service.stop()
            rpc_server.stop.assert_called_once_with()
            rpc_server.wait.assert_called_once_with()