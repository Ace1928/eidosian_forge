from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip_association
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
class TestDeleteLocalIPAssociation(TestLocalIPAssociation):

    def setUp(self):
        super().setUp()
        self._local_ip_association = network_fakes.create_local_ip_associations(count=2, attrs={'local_ip_id': self.local_ip.id})
        self.network_client.delete_local_ip_association = mock.Mock(return_value=None)
        self.network_client.find_local_ip = mock.Mock(return_value=self.local_ip)
        self.cmd = local_ip_association.DeleteLocalIPAssociation(self.app, self.namespace)

    def test_local_ip_association_delete(self):
        arglist = [self.local_ip.id, self._local_ip_association[0].fixed_port_id]
        verifylist = [('local_ip', self.local_ip.id), ('fixed_port_id', [self._local_ip_association[0].fixed_port_id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_local_ip_association.assert_called_once_with(self.local_ip.id, self._local_ip_association[0].fixed_port_id, ignore_missing=False)
        self.assertIsNone(result)

    def test_multi_local_ip_associations_delete(self):
        arglist = []
        fixed_port_id = []
        arglist.append(str(self.local_ip))
        for a in self._local_ip_association:
            arglist.append(a.fixed_port_id)
            fixed_port_id.append(a.fixed_port_id)
        verifylist = [('local_ip', str(self.local_ip)), ('fixed_port_id', fixed_port_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for a in self._local_ip_association:
            calls.append(call(a.local_ip_id, a.fixed_port_id, ignore_missing=False))
        self.network_client.delete_local_ip_association.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_local_ip_association_delete_with_exception(self):
        arglist = [self.local_ip.id, self._local_ip_association[0].fixed_port_id, 'unexist_fixed_port_id']
        verifylist = [('local_ip', self.local_ip.id), ('fixed_port_id', [self._local_ip_association[0].fixed_port_id, 'unexist_fixed_port_id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        delete_mock_result = [None, exceptions.CommandError]
        self.network_client.delete_local_ip_association = mock.MagicMock(side_effect=delete_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 Local IP Associations failed to delete.', str(e))
        self.network_client.delete_local_ip_association.assert_any_call(self.local_ip.id, 'unexist_fixed_port_id', ignore_missing=False)
        self.network_client.delete_local_ip_association.assert_any_call(self.local_ip.id, self._local_ip_association[0].fixed_port_id, ignore_missing=False)