import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestCreateBaremetalAllocation(TestBaremetalAllocation):

    def setUp(self):
        super(TestCreateBaremetalAllocation, self).setUp()
        self.baremetal_mock.allocation.create.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.ALLOCATION), loaded=True)
        self.baremetal_mock.allocation.wait.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.ALLOCATION), loaded=True)
        self.cmd = baremetal_allocation.CreateBaremetalAllocation(self.app, None)

    def test_baremetal_allocation_create(self):
        arglist = ['--resource-class', baremetal_fakes.baremetal_resource_class]
        verifylist = [('resource_class', baremetal_fakes.baremetal_resource_class)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'resource_class': baremetal_fakes.baremetal_resource_class}
        self.baremetal_mock.allocation.create.assert_called_once_with(**args)

    def test_baremetal_allocation_create_wait(self):
        arglist = ['--resource-class', baremetal_fakes.baremetal_resource_class, '--wait']
        verifylist = [('resource_class', baremetal_fakes.baremetal_resource_class), ('wait_timeout', 0)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'resource_class': baremetal_fakes.baremetal_resource_class}
        self.baremetal_mock.allocation.create.assert_called_once_with(**args)
        self.baremetal_mock.allocation.wait.assert_called_once_with(baremetal_fakes.ALLOCATION['uuid'], timeout=0)

    def test_baremetal_allocation_create_wait_with_timeout(self):
        arglist = ['--resource-class', baremetal_fakes.baremetal_resource_class, '--wait', '3600']
        verifylist = [('resource_class', baremetal_fakes.baremetal_resource_class), ('wait_timeout', 3600)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'resource_class': baremetal_fakes.baremetal_resource_class}
        self.baremetal_mock.allocation.create.assert_called_once_with(**args)
        self.baremetal_mock.allocation.wait.assert_called_once_with(baremetal_fakes.ALLOCATION['uuid'], timeout=3600)

    def test_baremetal_allocation_create_name_extras(self):
        arglist = ['--resource-class', baremetal_fakes.baremetal_resource_class, '--uuid', baremetal_fakes.baremetal_uuid, '--name', baremetal_fakes.baremetal_name, '--extra', 'key1=value1', '--extra', 'key2=value2']
        verifylist = [('resource_class', baremetal_fakes.baremetal_resource_class), ('uuid', baremetal_fakes.baremetal_uuid), ('name', baremetal_fakes.baremetal_name), ('extra', ['key1=value1', 'key2=value2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'resource_class': baremetal_fakes.baremetal_resource_class, 'uuid': baremetal_fakes.baremetal_uuid, 'name': baremetal_fakes.baremetal_name, 'extra': {'key1': 'value1', 'key2': 'value2'}}
        self.baremetal_mock.allocation.create.assert_called_once_with(**args)

    def test_baremetal_allocation_create_nodes_and_traits(self):
        arglist = ['--resource-class', baremetal_fakes.baremetal_resource_class, '--candidate-node', 'node1', '--trait', 'CUSTOM_1', '--candidate-node', 'node2', '--trait', 'CUSTOM_2']
        verifylist = [('resource_class', baremetal_fakes.baremetal_resource_class), ('candidate_nodes', ['node1', 'node2']), ('traits', ['CUSTOM_1', 'CUSTOM_2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'resource_class': baremetal_fakes.baremetal_resource_class, 'candidate_nodes': ['node1', 'node2'], 'traits': ['CUSTOM_1', 'CUSTOM_2']}
        self.baremetal_mock.allocation.create.assert_called_once_with(**args)

    def test_baremetal_allocation_create_owner(self):
        arglist = ['--resource-class', baremetal_fakes.baremetal_resource_class, '--owner', baremetal_fakes.baremetal_owner]
        verifylist = [('resource_class', baremetal_fakes.baremetal_resource_class), ('owner', baremetal_fakes.baremetal_owner)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'resource_class': baremetal_fakes.baremetal_resource_class, 'owner': baremetal_fakes.baremetal_owner}
        self.baremetal_mock.allocation.create.assert_called_once_with(**args)

    def test_baremetal_allocation_create_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.ClientException, self.cmd.take_action, parsed_args)

    def test_baremetal_allocation_backfill(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid]
        verifylist = [('node', baremetal_fakes.baremetal_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'node': baremetal_fakes.baremetal_uuid}
        self.baremetal_mock.allocation.create.assert_called_once_with(**args)