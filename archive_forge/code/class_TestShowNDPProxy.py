from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import ndp_proxy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowNDPProxy(TestNDPProxy):

    def setUp(self):
        super(TestShowNDPProxy, self).setUp()
        attrs = {'router_id': self.router.id, 'port_id': self.port.id}
        self.ndp_proxy = network_fakes.create_one_ndp_proxy(attrs)
        self.columns = ('created_at', 'description', 'id', 'ip_address', 'name', 'port_id', 'project_id', 'revision_number', 'router_id', 'updated_at')
        self.data = (self.ndp_proxy.created_at, self.ndp_proxy.description, self.ndp_proxy.id, self.ndp_proxy.ip_address, self.ndp_proxy.name, self.ndp_proxy.port_id, self.ndp_proxy.project_id, self.ndp_proxy.revision_number, self.ndp_proxy.router_id, self.ndp_proxy.updated_at)
        self.network_client.get_ndp_proxy = mock.Mock(return_value=self.ndp_proxy)
        self.network_client.find_ndp_proxy = mock.Mock(return_value=self.ndp_proxy)
        self.cmd = ndp_proxy.ShowNDPProxy(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_default_options(self):
        arglist = [self.ndp_proxy.id]
        verifylist = [('ndp_proxy', self.ndp_proxy.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_ndp_proxy.assert_called_once_with(self.ndp_proxy.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)