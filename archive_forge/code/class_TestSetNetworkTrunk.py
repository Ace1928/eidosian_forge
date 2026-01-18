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
class TestSetNetworkTrunk(TestNetworkTrunk):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    trunk_networks = network_fakes.create_networks(count=2)
    parent_port = network_fakes.create_one_port(attrs={'project_id': project.id, 'network_id': trunk_networks[0]['id']})
    sub_port = network_fakes.create_one_port(attrs={'project_id': project.id, 'network_id': trunk_networks[1]['id']})
    _trunk = network_fakes.create_one_trunk(attrs={'project_id': project.id, 'port_id': parent_port['id'], 'sub_ports': {'port_id': sub_port['id'], 'segmentation_id': 42, 'segmentation_type': 'vlan'}})
    columns = ('admin_state_up', 'id', 'name', 'description', 'port_id', 'project_id', 'status', 'sub_ports')
    data = (_trunk.id, _trunk.name, _trunk.description, _trunk.port_id, _trunk.project_id, _trunk.status, format_columns.ListDictColumn(_trunk.sub_ports))

    def setUp(self):
        super().setUp()
        self.network_client.update_trunk = mock.Mock(return_value=self._trunk)
        self.network_client.add_trunk_subports = mock.Mock(return_value=self._trunk)
        self.network_client.find_trunk = mock.Mock(return_value=self._trunk)
        self.network_client.find_port = mock.Mock(side_effect=[self.sub_port, self.sub_port])
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain
        self.cmd = network_trunk.SetNetworkTrunk(self.app, self.namespace)

    def _test_set_network_trunk_attr(self, attr, value):
        arglist = ['--%s' % attr, value, self._trunk[attr]]
        verifylist = [(attr, value), ('trunk', self._trunk[attr])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {attr: value}
        self.network_client.update_trunk.assert_called_once_with(self._trunk, **attrs)
        self.assertIsNone(result)

    def test_set_network_trunk_name(self):
        self._test_set_network_trunk_attr('name', 'trunky')

    def test_set_network_trunk_description(self):
        self._test_set_network_trunk_attr('description', 'description')

    def test_set_network_trunk_admin_state_up_disable(self):
        arglist = ['--disable', self._trunk['name']]
        verifylist = [('disable', True), ('trunk', self._trunk['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'admin_state_up': False}
        self.network_client.update_trunk.assert_called_once_with(self._trunk, **attrs)
        self.assertIsNone(result)

    def test_set_network_trunk_admin_state_up_enable(self):
        arglist = ['--enable', self._trunk['name']]
        verifylist = [('enable', True), ('trunk', self._trunk['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'admin_state_up': True}
        self.network_client.update_trunk.assert_called_once_with(self._trunk, **attrs)
        self.assertIsNone(result)

    def test_set_network_trunk_nothing(self):
        arglist = [self._trunk['name']]
        verifylist = [('trunk', self._trunk['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {}
        self.network_client.update_trunk.assert_called_once_with(self._trunk, **attrs)
        self.assertIsNone(result)

    def test_set_network_trunk_subports(self):
        subport = self._trunk['sub_ports'][0]
        arglist = ['--subport', 'port=%(port)s,segmentation-type=%(seg_type)s,segmentation-id=%(seg_id)s' % {'seg_id': subport['segmentation_id'], 'seg_type': subport['segmentation_type'], 'port': subport['port_id']}, self._trunk['name']]
        verifylist = [('trunk', self._trunk['name']), ('set_subports', [{'port': subport['port_id'], 'segmentation-id': str(subport['segmentation_id']), 'segmentation-type': subport['segmentation_type']}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.add_trunk_subports.assert_called_once_with(self._trunk, [subport])
        self.assertIsNone(result)

    def test_set_network_trunk_subports_without_optional_keys(self):
        subport = copy.copy(self._trunk['sub_ports'][0])
        subport.pop('segmentation_type')
        subport.pop('segmentation_id')
        arglist = ['--subport', 'port=%(port)s' % {'port': subport['port_id']}, self._trunk['name']]
        verifylist = [('trunk', self._trunk['name']), ('set_subports', [{'port': subport['port_id']}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.add_trunk_subports.assert_called_once_with(self._trunk, [subport])
        self.assertIsNone(result)

    def test_set_network_trunk_subports_without_required_key_fail(self):
        subport = self._trunk['sub_ports'][0]
        arglist = ['--subport', 'segmentation-type=%(seg_type)s,segmentation-id=%(seg_id)s' % {'seg_id': subport['segmentation_id'], 'seg_type': subport['segmentation_type']}, self._trunk['name']]
        verifylist = [('trunk', self._trunk['name']), ('set_subports', [{'segmentation-id': str(subport['segmentation_id']), 'segmentation-type': subport['segmentation_type']}])]
        with testtools.ExpectedException(test_utils.ParserException):
            self.check_parser(self.cmd, arglist, verifylist)
        self.network_client.add_trunk_subports.assert_not_called()

    def test_set_trunk_attrs_with_exception(self):
        arglist = ['--name', 'reallylongname', self._trunk['name']]
        verifylist = [('trunk', self._trunk['name']), ('name', 'reallylongname')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.network_client.update_trunk = mock.Mock(side_effect=exceptions.CommandError)
        with testtools.ExpectedException(exceptions.CommandError) as e:
            self.cmd.take_action(parsed_args)
            self.assertEqual("Failed to set trunk '%s': " % self._trunk['name'], str(e))
        attrs = {'name': 'reallylongname'}
        self.network_client.update_trunk.assert_called_once_with(self._trunk, **attrs)
        self.network_client.add_trunk_subports.assert_not_called()

    def test_set_trunk_add_subport_with_exception(self):
        arglist = ['--subport', 'port=invalid_subport', self._trunk['name']]
        verifylist = [('trunk', self._trunk['name']), ('set_subports', [{'port': 'invalid_subport'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.network_client.add_trunk_subports = mock.Mock(side_effect=exceptions.CommandError)
        self.network_client.find_port = mock.Mock(return_value={'id': 'invalid_subport'})
        with testtools.ExpectedException(exceptions.CommandError) as e:
            self.cmd.take_action(parsed_args)
            self.assertEqual("Failed to add subports to trunk '%s': " % self._trunk['name'], str(e))
        self.network_client.update_trunk.assert_called_once_with(self._trunk)
        self.network_client.add_trunk_subports.assert_called_once_with(self._trunk, [{'port_id': 'invalid_subport'}])