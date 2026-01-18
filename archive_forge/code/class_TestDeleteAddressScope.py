from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_scope
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteAddressScope(TestAddressScope):
    _address_scopes = network_fakes.create_address_scopes(count=2)

    def setUp(self):
        super(TestDeleteAddressScope, self).setUp()
        self.network_client.delete_address_scope = mock.Mock(return_value=None)
        self.network_client.find_address_scope = network_fakes.get_address_scopes(address_scopes=self._address_scopes)
        self.cmd = address_scope.DeleteAddressScope(self.app, self.namespace)

    def test_address_scope_delete(self):
        arglist = [self._address_scopes[0].name]
        verifylist = [('address_scope', [self._address_scopes[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.find_address_scope.assert_called_once_with(self._address_scopes[0].name, ignore_missing=False)
        self.network_client.delete_address_scope.assert_called_once_with(self._address_scopes[0])
        self.assertIsNone(result)

    def test_multi_address_scopes_delete(self):
        arglist = []
        verifylist = []
        for a in self._address_scopes:
            arglist.append(a.name)
        verifylist = [('address_scope', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for a in self._address_scopes:
            calls.append(call(a))
        self.network_client.delete_address_scope.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_address_scopes_delete_with_exception(self):
        arglist = [self._address_scopes[0].name, 'unexist_address_scope']
        verifylist = [('address_scope', [self._address_scopes[0].name, 'unexist_address_scope'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._address_scopes[0], exceptions.CommandError]
        self.network_client.find_address_scope = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 address scopes failed to delete.', str(e))
        self.network_client.find_address_scope.assert_any_call(self._address_scopes[0].name, ignore_missing=False)
        self.network_client.find_address_scope.assert_any_call('unexist_address_scope', ignore_missing=False)
        self.network_client.delete_address_scope.assert_called_once_with(self._address_scopes[0])