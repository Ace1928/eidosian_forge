import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
class TestBaremetalList(TestBaremetal):

    def setUp(self):
        super(TestBaremetalList, self).setUp()
        self.baremetal_mock.node.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL), loaded=True)]
        self.cmd = baremetal_node.ListBaremetalNode(self.app, None)

    def test_baremetal_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Name', 'Instance UUID', 'Power State', 'Provisioning State', 'Maintenance')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_instance_uuid, baremetal_fakes.baremetal_power_state, baremetal_fakes.baremetal_provision_state, baremetal_fakes.baremetal_maintenance),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': True, 'marker': None, 'limit': None}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)
        collist = ['Allocation UUID', 'Automated Clean', 'BIOS Interface', 'Boot Interface', 'Boot Mode', 'Chassis UUID', 'Clean Step', 'Conductor', 'Conductor Group', 'Console Enabled', 'Console Interface', 'Created At', 'Current RAID configuration', 'Deploy Interface', 'Deploy Step', 'Description', 'Driver', 'Driver Info', 'Driver Internal Info', 'Extra', 'Fault', 'Firmware Interface', 'Inspect Interface', 'Inspection Finished At', 'Inspection Started At', 'Instance Info', 'Instance UUID', 'Last Error', 'Lessee', 'Maintenance', 'Maintenance Reason', 'Management Interface', 'Name', 'Network Configuration', 'Network Interface', 'Owner', 'Parent Node', 'Power Interface', 'Power State', 'Properties', 'Protected', 'Protected Reason', 'Provision Updated At', 'Provisioning State', 'RAID Interface', 'Rescue Interface', 'Reservation', 'Resource Class', 'Retired', 'Retired Reason', 'Secure Boot', 'Storage Interface', 'Target Power State', 'Target Provision State', 'Target RAID configuration', 'Traits', 'UUID', 'Updated At', 'Vendor Interface']
        collist.sort()
        self.assertEqual(tuple(collist), columns)
        fake_values = {'Instance UUID': baremetal_fakes.baremetal_instance_uuid, 'Maintenance': baremetal_fakes.baremetal_maintenance, 'Name': baremetal_fakes.baremetal_name, 'Power State': baremetal_fakes.baremetal_power_state, 'Provisioning State': baremetal_fakes.baremetal_provision_state, 'UUID': baremetal_fakes.baremetal_uuid}
        values = tuple((fake_values.get(name, '') for name in collist))
        self.assertEqual((values,), tuple(data))

    def _test_baremetal_list_maintenance(self, maint_option, maint_value):
        arglist = [maint_option]
        verifylist = [('maintenance', maint_value)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'maintenance': maint_value}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_maintenance(self):
        self._test_baremetal_list_maintenance('--maintenance', True)

    def test_baremetal_list_no_maintenance(self):
        self._test_baremetal_list_maintenance('--no-maintenance', False)

    def test_baremetal_list_none_maintenance(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_both_maintenances(self):
        arglist = ['--maintenance', '--no-maintenance']
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def _test_baremetal_list_retired(self, retired_option, retired_value):
        arglist = [retired_option]
        verifylist = [('retired', retired_value)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'retired': retired_value}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_retired(self):
        self._test_baremetal_list_retired('--retired', True)

    def test_baremetal_list_no_retired(self):
        self._test_baremetal_list_retired('--no-retired', False)

    def test_baremetal_list_fault(self):
        arglist = ['--maintenance', '--fault', 'power failure']
        verifylist = [('maintenance', True), ('fault', 'power failure')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'maintenance': True, 'fault': 'power failure'}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_associated(self):
        arglist = ['--associated']
        verifylist = [('associated', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'associated': True}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_unassociated(self):
        arglist = ['--unassociated']
        verifylist = [('associated', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'associated': False}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_both_associated_unassociated_not_allowed(self):
        arglist = ['--associated', '--unassociated']
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_list_provision_state(self):
        arglist = ['--provision-state', 'active']
        verifylist = [('provision_state', 'active')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'provision_state': 'active'}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_driver(self):
        arglist = ['--driver', 'ipmi']
        verifylist = [('driver', 'ipmi')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'driver': 'ipmi'}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_resource_class(self):
        arglist = ['--resource-class', 'foo']
        verifylist = [('resource_class', 'foo')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'resource_class': 'foo'}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_chassis(self):
        chassis_uuid = 'aaaaaaaa-1111-bbbb-2222-cccccccccccc'
        arglist = ['--chassis', chassis_uuid]
        verifylist = [('chassis', chassis_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'chassis': chassis_uuid}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_conductor_group(self):
        conductor_group = 'in-the-closet-to-the-left'
        arglist = ['--conductor-group', conductor_group]
        verifylist = [('conductor_group', conductor_group)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'conductor_group': conductor_group}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_empty_conductor_group(self):
        conductor_group = ''
        arglist = ['--conductor-group', conductor_group]
        verifylist = [('conductor_group', conductor_group)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'conductor_group': conductor_group}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_by_conductor(self):
        conductor = 'fake-conductor'
        arglist = ['--conductor', conductor]
        verifylist = [('conductor', conductor)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'conductor': conductor}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_by_owner(self):
        owner = 'owner 1'
        arglist = ['--owner', owner]
        verifylist = [('owner', owner)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'owner': owner}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_has_description(self):
        description = 'there is no spoon'
        arglist = ['--description-contains', description]
        verifylist = [('description_contains', description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'description_contains': description}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_by_lessee(self):
        lessee = 'lessee 1'
        arglist = ['--lessee', lessee]
        verifylist = [('lessee', lessee)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'lessee': lessee}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_fields(self):
        arglist = ['--fields', 'uuid', 'name']
        verifylist = [('fields', [['uuid', 'name']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'name')}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_fields_multiple(self):
        arglist = ['--fields', 'uuid', 'name', '--fields', 'extra']
        verifylist = [('fields', [['uuid', 'name'], ['extra']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args=parsed_args)
        kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'name', 'extra')}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_invalid_fields(self):
        arglist = ['--fields', 'uuid', 'invalid']
        verifylist = [('fields', [['uuid', 'invalid']])]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_list_by_shards(self):
        arglist = ['--shards', 'myshard1', 'myshard2']
        verifylist = [('shards', ['myshard1', 'myshard2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'shards': ['myshard1', 'myshard2']}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_sharded(self):
        arglist = ['--sharded']
        verifylist = [('sharded', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'sharded': True}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_unsharded(self):
        arglist = ['--unsharded']
        verifylist = [('sharded', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'sharded': False}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_sharded_unsharded_fail(self):
        arglist = ['--sharded', '--unsharded']
        verifylist = [('sharded', True), ('sharded', False)]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_list_by_parent_node(self):
        parent_node = 'node1'
        arglist = ['--parent-node', parent_node]
        verifylist = [('parent_node', parent_node)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'parent_node': parent_node}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)

    def test_baremetal_list_include_children(self):
        arglist = ['--include-children']
        verifylist = [('include_children', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'include_children': True}
        self.baremetal_mock.node.list.assert_called_with(**kwargs)