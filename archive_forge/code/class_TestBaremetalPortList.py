import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalPortList(TestBaremetalPort):

    def setUp(self):
        super(TestBaremetalPortList, self).setUp()
        self.baremetal_mock.port.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_PORT), loaded=True)]
        self.cmd = baremetal_port.ListBaremetalPort(self.app, None)

    def test_baremetal_port_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.port.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Address')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_port_uuid, baremetal_fakes.baremetal_port_address),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_port_list_address(self):
        arglist = ['--address', baremetal_fakes.baremetal_port_address]
        verifylist = [('address', baremetal_fakes.baremetal_port_address)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'address': baremetal_fakes.baremetal_port_address, 'marker': None, 'limit': None}
        self.baremetal_mock.port.list.assert_called_with(**kwargs)

    def test_baremetal_port_list_node(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid]
        verifylist = [('node', baremetal_fakes.baremetal_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'node': baremetal_fakes.baremetal_uuid, 'marker': None, 'limit': None}
        self.baremetal_mock.port.list.assert_called_with(**kwargs)

    def test_baremetal_port_list_portgroup(self):
        arglist = ['--port-group', baremetal_fakes.baremetal_portgroup_uuid]
        verifylist = [('portgroup', baremetal_fakes.baremetal_portgroup_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'portgroup': baremetal_fakes.baremetal_portgroup_uuid, 'marker': None, 'limit': None}
        self.baremetal_mock.port.list.assert_called_with(**kwargs)

    def test_baremetal_port_list_long(self):
        arglist = ['--long']
        verifylist = [('detail', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': True, 'marker': None, 'limit': None}
        self.baremetal_mock.port.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Address', 'Created At', 'Extra', 'Node UUID', 'Local Link Connection', 'Portgroup UUID', 'PXE boot enabled', 'Physical Network', 'Updated At', 'Internal Info', 'Is Smart NIC port')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_port_uuid, baremetal_fakes.baremetal_port_address, '', oscutils.format_dict(baremetal_fakes.baremetal_port_extra), baremetal_fakes.baremetal_uuid, '', '', '', '', '', '', ''),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_port_list_fields(self):
        arglist = ['--fields', 'uuid', 'address']
        verifylist = [('fields', [['uuid', 'address']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'address')}
        self.baremetal_mock.port.list.assert_called_with(**kwargs)

    def test_baremetal_port_list_fields_multiple(self):
        arglist = ['--fields', 'uuid', 'address', '--fields', 'extra']
        verifylist = [('fields', [['uuid', 'address'], ['extra']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None, 'detail': False, 'fields': ('uuid', 'address', 'extra')}
        self.baremetal_mock.port.list.assert_called_with(**kwargs)

    def test_baremetal_port_list_invalid_fields(self):
        arglist = ['--fields', 'uuid', 'invalid']
        verifylist = [('fields', [['uuid', 'invalid']])]
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)