from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetAddressGroup(TestAddressGroup):
    _address_group = network_fakes.create_one_address_group()

    def setUp(self):
        super(TestSetAddressGroup, self).setUp()
        self.network_client.update_address_group = mock.Mock(return_value=None)
        self.network_client.find_address_group = mock.Mock(return_value=self._address_group)
        self.network_client.add_addresses_to_address_group = mock.Mock(return_value=self._address_group)
        self.cmd = address_group.SetAddressGroup(self.app, self.namespace)

    def test_set_nothing(self):
        arglist = [self._address_group.name]
        verifylist = [('address_group', self._address_group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_address_group.assert_not_called()
        self.network_client.add_addresses_to_address_group.assert_not_called()
        self.assertIsNone(result)

    def test_set_name_and_description(self):
        arglist = ['--name', 'new_address_group_name', '--description', 'new_address_group_description', self._address_group.name]
        verifylist = [('name', 'new_address_group_name'), ('description', 'new_address_group_description'), ('address_group', self._address_group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'new_address_group_name', 'description': 'new_address_group_description'}
        self.network_client.update_address_group.assert_called_with(self._address_group, **attrs)
        self.assertIsNone(result)

    def test_set_one_address(self):
        arglist = [self._address_group.name, '--address', '10.0.0.2']
        verifylist = [('address_group', self._address_group.name), ('address', ['10.0.0.2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.add_addresses_to_address_group.assert_called_once_with(self._address_group, ['10.0.0.2/32'])
        self.assertIsNone(result)

    def test_set_multiple_addresses(self):
        arglist = [self._address_group.name, '--address', '10.0.0.2', '--address', '2001::/16']
        verifylist = [('address_group', self._address_group.name), ('address', ['10.0.0.2', '2001::/16'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.add_addresses_to_address_group.assert_called_once_with(self._address_group, ['10.0.0.2/32', '2001::/16'])
        self.assertIsNone(result)