from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowNetworkFlavor(TestNetworkFlavor):
    new_network_flavor = network_fakes.create_one_network_flavor()
    columns = ('description', 'enabled', 'id', 'name', 'service_type', 'service_profile_ids')
    data = (new_network_flavor.description, new_network_flavor.is_enabled, new_network_flavor.id, new_network_flavor.name, new_network_flavor.service_type, new_network_flavor.service_profile_ids)

    def setUp(self):
        super(TestShowNetworkFlavor, self).setUp()
        self.network_client.find_flavor = mock.Mock(return_value=self.new_network_flavor)
        self.cmd = network_flavor.ShowNetworkFlavor(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self.new_network_flavor.name]
        verifylist = [('flavor', self.new_network_flavor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_flavor.assert_called_once_with(self.new_network_flavor.name, ignore_missing=False)
        self.assertEqual(set(self.columns), set(columns))
        self.assertEqual(set(self.data), set(data))