from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteNetworkFlavor(TestNetworkFlavor):
    _network_flavors = network_fakes.create_flavor(count=2)

    def setUp(self):
        super(TestDeleteNetworkFlavor, self).setUp()
        self.network_client.delete_flavor = mock.Mock(return_value=None)
        self.network_client.find_flavor = network_fakes.get_flavor(network_flavors=self._network_flavors)
        self.cmd = network_flavor.DeleteNetworkFlavor(self.app, self.namespace)

    def test_network_flavor_delete(self):
        arglist = [self._network_flavors[0].name]
        verifylist = [('flavor', [self._network_flavors[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.find_flavor.assert_called_once_with(self._network_flavors[0].name, ignore_missing=False)
        self.network_client.delete_flavor.assert_called_once_with(self._network_flavors[0])
        self.assertIsNone(result)

    def test_multi_network_flavors_delete(self):
        arglist = []
        verifylist = []
        for a in self._network_flavors:
            arglist.append(a.name)
        verifylist = [('flavor', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for a in self._network_flavors:
            calls.append(mock.call(a))
        self.network_client.delete_flavor.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_network_flavors_delete_with_exception(self):
        arglist = [self._network_flavors[0].name, 'unexist_network_flavor']
        verifylist = [('flavor', [self._network_flavors[0].name, 'unexist_network_flavor'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._network_flavors[0], exceptions.CommandError]
        self.network_client.find_flavor = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 flavors failed to delete.', str(e))
        self.network_client.find_flavor.assert_any_call(self._network_flavors[0].name, ignore_missing=False)
        self.network_client.find_flavor.assert_any_call('unexist_network_flavor', ignore_missing=False)
        self.network_client.delete_flavor.assert_called_once_with(self._network_flavors[0])