import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestCreateBaremetalPortGroup(TestBaremetalPortGroup):

    def setUp(self):
        super(TestCreateBaremetalPortGroup, self).setUp()
        self.baremetal_mock.portgroup.create.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.PORTGROUP), loaded=True)
        self.cmd = baremetal_portgroup.CreateBaremetalPortGroup(self.app, None)

    def test_baremetal_portgroup_create(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'node_uuid': baremetal_fakes.baremetal_uuid}
        self.baremetal_mock.portgroup.create.assert_called_once_with(**args)

    def test_baremetal_portgroup_create_name_address_uuid(self):
        arglist = ['--address', baremetal_fakes.baremetal_portgroup_address, '--node', baremetal_fakes.baremetal_uuid, '--name', baremetal_fakes.baremetal_portgroup_name, '--uuid', baremetal_fakes.baremetal_portgroup_uuid]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_portgroup_address), ('name', baremetal_fakes.baremetal_portgroup_name), ('uuid', baremetal_fakes.baremetal_portgroup_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'address': baremetal_fakes.baremetal_portgroup_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'name': baremetal_fakes.baremetal_portgroup_name, 'uuid': baremetal_fakes.baremetal_portgroup_uuid}
        self.baremetal_mock.portgroup.create.assert_called_once_with(**args)

    def test_baremetal_portgroup_create_support_standalone_ports(self):
        arglist = ['--address', baremetal_fakes.baremetal_portgroup_address, '--node', baremetal_fakes.baremetal_uuid, '--support-standalone-ports']
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_portgroup_address), ('support_standalone_ports', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'address': baremetal_fakes.baremetal_portgroup_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'standalone_ports_supported': True}
        self.baremetal_mock.portgroup.create.assert_called_once_with(**args)

    def test_baremetal_portgroup_create_unsupport_standalone_ports(self):
        arglist = ['--address', baremetal_fakes.baremetal_portgroup_address, '--node', baremetal_fakes.baremetal_uuid, '--unsupport-standalone-ports']
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_portgroup_address), ('unsupport_standalone_ports', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'address': baremetal_fakes.baremetal_portgroup_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'standalone_ports_supported': False}
        self.baremetal_mock.portgroup.create.assert_called_once_with(**args)

    def test_baremetal_portgroup_create_name_extras(self):
        arglist = ['--address', baremetal_fakes.baremetal_portgroup_address, '--node', baremetal_fakes.baremetal_uuid, '--name', baremetal_fakes.baremetal_portgroup_name, '--extra', 'key1=value1', '--extra', 'key2=value2']
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('address', baremetal_fakes.baremetal_portgroup_address), ('name', baremetal_fakes.baremetal_portgroup_name), ('extra', ['key1=value1', 'key2=value2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'address': baremetal_fakes.baremetal_portgroup_address, 'node_uuid': baremetal_fakes.baremetal_uuid, 'name': baremetal_fakes.baremetal_portgroup_name, 'extra': baremetal_fakes.baremetal_portgroup_extra}
        self.baremetal_mock.portgroup.create.assert_called_once_with(**args)

    def test_baremetal_portgroup_create_mode_properties(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--mode', baremetal_fakes.baremetal_portgroup_mode, '--property', 'key1=value11', '--property', 'key2=value22']
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('mode', baremetal_fakes.baremetal_portgroup_mode), ('properties', ['key1=value11', 'key2=value22'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'node_uuid': baremetal_fakes.baremetal_uuid, 'mode': baremetal_fakes.baremetal_portgroup_mode, 'properties': baremetal_fakes.baremetal_portgroup_properties}
        self.baremetal_mock.portgroup.create.assert_called_once_with(**args)

    def test_baremetal_portgroup_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)