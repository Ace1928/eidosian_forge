from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip_association
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
class TestCreateLocalIPAssociation(TestLocalIPAssociation):

    def setUp(self):
        super().setUp()
        self.new_local_ip_association = network_fakes.create_one_local_ip_association(attrs={'fixed_port_id': self.fixed_port.id, 'local_ip_id': self.local_ip.id})
        self.network_client.create_local_ip_association = mock.Mock(return_value=self.new_local_ip_association)
        self.network_client.find_local_ip = mock.Mock(return_value=self.local_ip)
        self.cmd = local_ip_association.CreateLocalIPAssociation(self.app, self.namespace)
        self.columns = ('local_ip_address', 'fixed_port_id', 'fixed_ip', 'host')
        self.data = (self.new_local_ip_association.local_ip_address, self.new_local_ip_association.fixed_port_id, self.new_local_ip_association.fixed_ip, self.new_local_ip_association.host)

    def test_create_no_options(self):
        arglist = [self.new_local_ip_association.local_ip_id, self.new_local_ip_association.fixed_port_id]
        verifylist = [('local_ip', self.new_local_ip_association.local_ip_id), ('fixed_port', self.new_local_ip_association.fixed_port_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_local_ip_association.assert_called_once_with(self.new_local_ip_association.local_ip_id, **{'fixed_port_id': self.new_local_ip_association.fixed_port_id})
        self.assertEqual(set(self.columns), set(columns))
        self.assertEqual(set(self.data), set(data))

    def test_create_all_options(self):
        arglist = [self.new_local_ip_association.local_ip_id, self.new_local_ip_association.fixed_port_id, '--fixed-ip', self.new_local_ip_association.fixed_ip]
        verifylist = [('local_ip', self.new_local_ip_association.local_ip_id), ('fixed_port', self.new_local_ip_association.fixed_port_id), ('fixed_ip', self.new_local_ip_association.fixed_ip)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_local_ip_association.assert_called_once_with(self.new_local_ip_association.local_ip_id, **{'fixed_port_id': self.new_local_ip_association.fixed_port_id, 'fixed_ip': self.new_local_ip_association.fixed_ip})
        self.assertEqual(set(self.columns), set(columns))
        self.assertEqual(set(self.data), set(data))