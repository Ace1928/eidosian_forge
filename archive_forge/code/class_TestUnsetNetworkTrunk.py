import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestUnsetNetworkTrunk(TestNetworkTrunk):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    trunk_networks = network_fakes.create_networks(count=2)
    parent_port = network_fakes.create_one_port(attrs={'project_id': project.id, 'network_id': trunk_networks[0]['id']})
    sub_port = network_fakes.create_one_port(attrs={'project_id': project.id, 'network_id': trunk_networks[1]['id']})
    _trunk = network_fakes.create_one_trunk(attrs={'project_id': project.id, 'port_id': parent_port['id'], 'sub_ports': {'port_id': sub_port['id'], 'segmentation_id': 42, 'segmentation_type': 'vlan'}})
    columns = ('admin_state_up', 'id', 'name', 'port_id', 'project_id', 'status', 'sub_ports')
    data = (network_trunk.AdminStateColumn(_trunk['admin_state_up']), _trunk['id'], _trunk['name'], _trunk['port_id'], _trunk['project_id'], _trunk['status'], format_columns.ListDictColumn(_trunk['sub_ports']))

    def setUp(self):
        super().setUp()
        self.network_client.find_trunk = mock.Mock(return_value=self._trunk)
        self.network_client.find_port = mock.Mock(side_effect=[self.sub_port, self.sub_port])
        self.network_client.delete_trunk_subports = mock.Mock(return_value=None)
        self.cmd = network_trunk.UnsetNetworkTrunk(self.app, self.namespace)

    def test_unset_network_trunk_subport(self):
        subport = self._trunk['sub_ports'][0]
        arglist = ['--subport', subport['port_id'], self._trunk['name']]
        verifylist = [('trunk', self._trunk['name']), ('unset_subports', [subport['port_id']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_trunk_subports.assert_called_once_with(self._trunk, [{'port_id': subport['port_id']}])
        self.assertIsNone(result)

    def test_unset_subport_no_arguments_fail(self):
        arglist = [self._trunk['name']]
        verifylist = [('trunk', self._trunk['name'])]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)