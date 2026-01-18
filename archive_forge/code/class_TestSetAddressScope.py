from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_scope
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetAddressScope(TestAddressScope):
    _address_scope = network_fakes.create_one_address_scope()

    def setUp(self):
        super(TestSetAddressScope, self).setUp()
        self.network_client.update_address_scope = mock.Mock(return_value=None)
        self.network_client.find_address_scope = mock.Mock(return_value=self._address_scope)
        self.cmd = address_scope.SetAddressScope(self.app, self.namespace)

    def test_set_nothing(self):
        arglist = [self._address_scope.name]
        verifylist = [('address_scope', self._address_scope.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {}
        self.network_client.update_address_scope.assert_called_with(self._address_scope, **attrs)
        self.assertIsNone(result)

    def test_set_name_and_share(self):
        arglist = ['--name', 'new_address_scope', '--share', self._address_scope.name]
        verifylist = [('name', 'new_address_scope'), ('share', True), ('address_scope', self._address_scope.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'new_address_scope', 'shared': True}
        self.network_client.update_address_scope.assert_called_with(self._address_scope, **attrs)
        self.assertIsNone(result)

    def test_set_no_share(self):
        arglist = ['--no-share', self._address_scope.name]
        verifylist = [('no_share', True), ('address_scope', self._address_scope.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'shared': False}
        self.network_client.update_address_scope.assert_called_with(self._address_scope, **attrs)
        self.assertIsNone(result)