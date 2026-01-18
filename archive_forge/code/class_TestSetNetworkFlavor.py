from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetNetworkFlavor(TestNetworkFlavor):
    new_network_flavor = network_fakes.create_one_network_flavor()

    def setUp(self):
        super(TestSetNetworkFlavor, self).setUp()
        self.network_client.update_flavor = mock.Mock(return_value=None)
        self.network_client.find_flavor = mock.Mock(return_value=self.new_network_flavor)
        self.cmd = network_flavor.SetNetworkFlavor(self.app, self.namespace)

    def test_set_nothing(self):
        arglist = [self.new_network_flavor.name]
        verifylist = [('flavor', self.new_network_flavor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {}
        self.network_client.update_flavor.assert_called_with(self.new_network_flavor, **attrs)
        self.assertIsNone(result)

    def test_set_name_and_enable(self):
        arglist = ['--name', 'new_network_flavor', '--enable', self.new_network_flavor.name]
        verifylist = [('name', 'new_network_flavor'), ('enable', True), ('flavor', self.new_network_flavor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'new_network_flavor', 'enabled': True}
        self.network_client.update_flavor.assert_called_with(self.new_network_flavor, **attrs)
        self.assertIsNone(result)

    def test_set_disable(self):
        arglist = ['--disable', self.new_network_flavor.name]
        verifylist = [('disable', True), ('flavor', self.new_network_flavor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'enabled': False}
        self.network_client.update_flavor.assert_called_with(self.new_network_flavor, **attrs)
        self.assertIsNone(result)