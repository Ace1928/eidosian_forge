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
class TestCreateNetworkTrunk(TestNetworkTrunk):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    trunk_networks = network_fakes.create_networks(count=2)
    parent_port = network_fakes.create_one_port(attrs={'project_id': project.id, 'network_id': trunk_networks[0]['id']})
    sub_port = network_fakes.create_one_port(attrs={'project_id': project.id, 'network_id': trunk_networks[1]['id']})
    new_trunk = network_fakes.create_one_trunk(attrs={'project_id': project.id, 'port_id': parent_port['id'], 'sub_ports': {'port_id': sub_port['id'], 'segmentation_id': 42, 'segmentation_type': 'vlan'}})
    columns = ('description', 'id', 'is_admin_state_up', 'name', 'port_id', 'project_id', 'status', 'sub_ports', 'tags')
    data = (new_trunk.description, new_trunk.id, new_trunk.is_admin_state_up, new_trunk.name, new_trunk.port_id, new_trunk.project_id, new_trunk.status, format_columns.ListDictColumn(new_trunk.sub_ports), [])

    def setUp(self):
        super().setUp()
        self.network_client.create_trunk = mock.Mock(return_value=self.new_trunk)
        self.network_client.find_port = mock.Mock(side_effect=[self.parent_port, self.sub_port])
        self.cmd = network_trunk.CreateNetworkTrunk(self.app, self.namespace)
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = ['--parent-port', self.new_trunk['port_id'], self.new_trunk['name']]
        verifylist = [('parent_port', self.new_trunk['port_id']), ('name', self.new_trunk['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_trunk.assert_called_once_with(**{'name': self.new_trunk['name'], 'admin_state_up': self.new_trunk['admin_state_up'], 'port_id': self.new_trunk['port_id']})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_full_options(self):
        self.new_trunk['description'] = 'foo description'
        subport = self.new_trunk.sub_ports[0]
        arglist = ['--disable', '--description', self.new_trunk.description, '--parent-port', self.new_trunk.port_id, '--subport', 'port=%(port)s,segmentation-type=%(seg_type)s,segmentation-id=%(seg_id)s' % {'seg_id': subport['segmentation_id'], 'seg_type': subport['segmentation_type'], 'port': subport['port_id']}, self.new_trunk.name]
        verifylist = [('name', self.new_trunk.name), ('description', self.new_trunk.description), ('parent_port', self.new_trunk.port_id), ('add_subports', [{'port': subport['port_id'], 'segmentation-id': str(subport['segmentation_id']), 'segmentation-type': subport['segmentation_type']}]), ('disable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_trunk.assert_called_once_with(**{'name': self.new_trunk.name, 'description': self.new_trunk.description, 'admin_state_up': False, 'port_id': self.new_trunk.port_id, 'sub_ports': [subport]})
        self.assertEqual(self.columns, columns)
        data_with_desc = list(self.data)
        data_with_desc[0] = self.new_trunk['description']
        data_with_desc = tuple(data_with_desc)
        self.assertEqual(data_with_desc, data)

    def test_create_trunk_with_subport_invalid_segmentation_id_fail(self):
        subport = self.new_trunk.sub_ports[0]
        arglist = ['--parent-port', self.new_trunk.port_id, '--subport', 'port=%(port)s,segmentation-type=%(seg_type)s,segmentation-id=boom' % {'seg_type': subport['segmentation_type'], 'port': subport['port_id']}, self.new_trunk.name]
        verifylist = [('name', self.new_trunk.name), ('parent_port', self.new_trunk.port_id), ('add_subports', [{'port': subport['port_id'], 'segmentation-id': 'boom', 'segmentation-type': subport['segmentation_type']}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with testtools.ExpectedException(exceptions.CommandError) as e:
            self.cmd.take_action(parsed_args)
            self.assertEqual("Segmentation-id 'boom' is not an integer", str(e))

    def test_create_network_trunk_subports_without_optional_keys(self):
        subport = copy.copy(self.new_trunk.sub_ports[0])
        subport.pop('segmentation_type')
        subport.pop('segmentation_id')
        arglist = ['--parent-port', self.new_trunk.port_id, '--subport', 'port=%(port)s' % {'port': subport['port_id']}, self.new_trunk.name]
        verifylist = [('name', self.new_trunk.name), ('parent_port', self.new_trunk.port_id), ('add_subports', [{'port': subport['port_id']}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_trunk.assert_called_once_with(**{'name': self.new_trunk.name, 'admin_state_up': True, 'port_id': self.new_trunk.port_id, 'sub_ports': [subport]})
        self.assertEqual(self.columns, columns)
        data_with_desc = list(self.data)
        data_with_desc[0] = self.new_trunk['description']
        data_with_desc = tuple(data_with_desc)
        self.assertEqual(data_with_desc, data)

    def test_create_network_trunk_subports_without_required_key_fail(self):
        subport = self.new_trunk.sub_ports[0]
        arglist = ['--parent-port', self.new_trunk.port_id, '--subport', 'segmentation-type=%(seg_type)s,segmentation-id=%(seg_id)s' % {'seg_id': subport['segmentation_id'], 'seg_type': subport['segmentation_type']}, self.new_trunk.name]
        verifylist = [('name', self.new_trunk.name), ('parent_port', self.new_trunk.port_id), ('add_subports', [{'segmentation_id': str(subport['segmentation_id']), 'segmentation_type': subport['segmentation_type']}])]
        with testtools.ExpectedException(test_utils.ParserException):
            self.check_parser(self.cmd, arglist, verifylist)