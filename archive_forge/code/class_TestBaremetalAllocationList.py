import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalAllocationList(TestBaremetalAllocation):

    def setUp(self):
        super(TestBaremetalAllocationList, self).setUp()
        self.baremetal_mock.allocation.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.ALLOCATION), loaded=True)]
        self.cmd = baremetal_allocation.ListBaremetalAllocation(self.app, None)

    def test_baremetal_allocation_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.allocation.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Name', 'Resource Class', 'State', 'Node UUID')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_resource_class, baremetal_fakes.baremetal_allocation_state, baremetal_fakes.baremetal_uuid),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_allocation_list_node(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid]
        verifylist = [('node', baremetal_fakes.baremetal_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'node': baremetal_fakes.baremetal_uuid, 'marker': None, 'limit': None}
        self.baremetal_mock.allocation.list.assert_called_once_with(**kwargs)
        collist = ('UUID', 'Name', 'Resource Class', 'State', 'Node UUID')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_resource_class, baremetal_fakes.baremetal_allocation_state, baremetal_fakes.baremetal_uuid),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_allocation_list_resource_class(self):
        arglist = ['--resource-class', baremetal_fakes.baremetal_resource_class]
        verifylist = [('resource_class', baremetal_fakes.baremetal_resource_class)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'resource_class': baremetal_fakes.baremetal_resource_class, 'marker': None, 'limit': None}
        self.baremetal_mock.allocation.list.assert_called_once_with(**kwargs)
        collist = ('UUID', 'Name', 'Resource Class', 'State', 'Node UUID')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_resource_class, baremetal_fakes.baremetal_allocation_state, baremetal_fakes.baremetal_uuid),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_allocation_list_owner(self):
        arglist = ['--owner', baremetal_fakes.baremetal_owner]
        verifylist = [('owner', baremetal_fakes.baremetal_owner)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'owner': baremetal_fakes.baremetal_owner, 'marker': None, 'limit': None}
        self.baremetal_mock.allocation.list.assert_called_once_with(**kwargs)
        collist = ('UUID', 'Name', 'Resource Class', 'State', 'Node UUID')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_resource_class, baremetal_fakes.baremetal_allocation_state, baremetal_fakes.baremetal_uuid),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_allocation_list_state(self):
        arglist = ['--state', baremetal_fakes.baremetal_allocation_state]
        verifylist = [('state', baremetal_fakes.baremetal_allocation_state)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'state': baremetal_fakes.baremetal_allocation_state, 'marker': None, 'limit': None}
        self.baremetal_mock.allocation.list.assert_called_once_with(**kwargs)
        collist = ('UUID', 'Name', 'Resource Class', 'State', 'Node UUID')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_resource_class, baremetal_fakes.baremetal_allocation_state, baremetal_fakes.baremetal_uuid),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_allocation_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.allocation.list.assert_called_once_with(**kwargs)
        collist = ('UUID', 'Name', 'State', 'Owner', 'Node UUID', 'Last Error', 'Resource Class', 'Traits', 'Candidate Nodes', 'Extra', 'Created At', 'Updated At')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_allocation_state, '', baremetal_fakes.baremetal_uuid, '', baremetal_fakes.baremetal_resource_class, '', '', '', '', ''),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_allocation_list_fields(self):
        arglist = ['--fields', 'uuid', 'node_uuid']
        verifylist = [('fields', [['uuid', 'node_uuid']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'fields': ('uuid', 'node_uuid')}
        self.baremetal_mock.allocation.list.assert_called_once_with(**kwargs)

    def test_baremetal_allocation_list_fields_multiple(self):
        arglist = ['--fields', 'uuid', 'node_uuid', '--fields', 'extra']
        verifylist = [('fields', [['uuid', 'node_uuid'], ['extra']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'fields': ('uuid', 'node_uuid', 'extra')}
        self.baremetal_mock.allocation.list.assert_called_once_with(**kwargs)

    def test_baremetal_allocation_list_invalid_fields(self):
        arglist = ['--fields', 'uuid', 'invalid']
        verifylist = [('fields', [['uuid', 'invalid']])]
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)