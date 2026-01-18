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
class TestBaremetalUnset(TestBaremetal):

    def setUp(self):
        super(TestBaremetalUnset, self).setUp()
        self.baremetal_mock.node.update.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL), loaded=True)
        self.cmd = baremetal_node.UnsetBaremetalNode(self.app, None)

    def test_baremetal_unset_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_unset_no_property(self):
        arglist = ['node_uuid']
        verifylist = [('nodes', ['node_uuid'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.assertFalse(self.baremetal_mock.node.update.called)

    def test_baremetal_unset_one_property(self):
        arglist = ['node_uuid', '--property', 'path/to/property']
        verifylist = [('nodes', ['node_uuid']), ('property', ['path/to/property'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/properties/path/to/property', 'op': 'remove'}])

    def test_baremetal_unset_multiple_properties(self):
        arglist = ['node_uuid', '--property', 'path/to/property', '--property', 'other/path']
        verifylist = [('nodes', ['node_uuid']), ('property', ['path/to/property', 'other/path'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/properties/path/to/property', 'op': 'remove'}, {'path': '/properties/other/path', 'op': 'remove'}])

    def test_baremetal_unset_instance_uuid(self):
        arglist = ['node_uuid', '--instance-uuid']
        verifylist = [('nodes', ['node_uuid']), ('instance_uuid', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/instance_uuid', 'op': 'remove'}])

    def test_baremetal_unset_name(self):
        arglist = ['node_uuid', '--name']
        verifylist = [('nodes', ['node_uuid']), ('name', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/name', 'op': 'remove'}])

    def test_baremetal_unset_resource_class(self):
        arglist = ['node_uuid', '--resource-class']
        verifylist = [('nodes', ['node_uuid']), ('resource_class', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/resource_class', 'op': 'remove'}])

    def test_baremetal_unset_conductor_group(self):
        arglist = ['node_uuid', '--conductor-group']
        verifylist = [('nodes', ['node_uuid']), ('conductor_group', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/conductor_group', 'op': 'remove'}])

    def test_baremetal_unset_automated_clean(self):
        arglist = ['node_uuid', '--automated-clean']
        verifylist = [('nodes', ['node_uuid']), ('automated_clean', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/automated_clean', 'op': 'remove'}])

    def test_baremetal_unset_protected(self):
        arglist = ['node_uuid', '--protected']
        verifylist = [('nodes', ['node_uuid']), ('protected', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/protected', 'op': 'remove'}])

    def test_baremetal_unset_protected_reason(self):
        arglist = ['node_uuid', '--protected-reason']
        verifylist = [('nodes', ['node_uuid']), ('protected_reason', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/protected_reason', 'op': 'remove'}])

    def test_baremetal_unset_retired(self):
        arglist = ['node_uuid', '--retired']
        verifylist = [('nodes', ['node_uuid']), ('retired', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/retired', 'op': 'remove'}])

    def test_baremetal_unset_retired_reason(self):
        arglist = ['node_uuid', '--retired-reason']
        verifylist = [('nodes', ['node_uuid']), ('retired_reason', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/retired_reason', 'op': 'remove'}])

    def test_baremetal_unset_extra(self):
        arglist = ['node_uuid', '--extra', 'foo']
        verifylist = [('nodes', ['node_uuid']), ('extra', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/extra/foo', 'op': 'remove'}])

    def test_baremetal_unset_driver_info(self):
        arglist = ['node_uuid', '--driver-info', 'foo']
        verifylist = [('nodes', ['node_uuid']), ('driver_info', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/driver_info/foo', 'op': 'remove'}])

    def test_baremetal_unset_instance_info(self):
        arglist = ['node_uuid', '--instance-info', 'foo']
        verifylist = [('nodes', ['node_uuid']), ('instance_info', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/instance_info/foo', 'op': 'remove'}])

    def test_baremetal_unset_target_raid_config(self):
        self.cmd.log = mock.Mock(autospec=True)
        arglist = ['node_uuid', '--target-raid-config']
        verifylist = [('nodes', ['node_uuid']), ('target_raid_config', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cmd.log.warning.assert_not_called()
        self.assertFalse(self.baremetal_mock.node.update.called)
        self.baremetal_mock.node.set_target_raid_config.assert_called_once_with('node_uuid', {})

    def test_baremetal_unset_target_raid_config_and_name(self):
        self.cmd.log = mock.Mock(autospec=True)
        arglist = ['node_uuid', '--name', '--target-raid-config']
        verifylist = [('nodes', ['node_uuid']), ('name', True), ('target_raid_config', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cmd.log.warning.assert_not_called()
        self.baremetal_mock.node.set_target_raid_config.assert_called_once_with('node_uuid', {})
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/name', 'op': 'remove'}])

    def test_baremetal_unset_chassis_uuid(self):
        arglist = ['node_uuid', '--chassis-uuid']
        verifylist = [('nodes', ['node_uuid']), ('chassis_uuid', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/chassis_uuid', 'op': 'remove'}])

    def _test_baremetal_unset_hw_interface(self, interface):
        arglist = ['node_uuid', '--%s-interface' % interface]
        verifylist = [('nodes', ['node_uuid']), ('%s_interface' % interface, True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/%s_interface' % interface, 'op': 'remove'}])

    def test_baremetal_unset_bios_interface(self):
        self._test_baremetal_unset_hw_interface('bios')

    def test_baremetal_unset_boot_interface(self):
        self._test_baremetal_unset_hw_interface('boot')

    def test_baremetal_unset_console_interface(self):
        self._test_baremetal_unset_hw_interface('console')

    def test_baremetal_unset_deploy_interface(self):
        self._test_baremetal_unset_hw_interface('deploy')

    def test_baremetal_unset_firmware_interface(self):
        self._test_baremetal_unset_hw_interface('firmware')

    def test_baremetal_unset_inspect_interface(self):
        self._test_baremetal_unset_hw_interface('inspect')

    def test_baremetal_unset_management_interface(self):
        self._test_baremetal_unset_hw_interface('management')

    def test_baremetal_unset_network_interface(self):
        self._test_baremetal_unset_hw_interface('network')

    def test_baremetal_unset_power_interface(self):
        self._test_baremetal_unset_hw_interface('power')

    def test_baremetal_unset_raid_interface(self):
        self._test_baremetal_unset_hw_interface('raid')

    def test_baremetal_unset_rescue_interface(self):
        self._test_baremetal_unset_hw_interface('rescue')

    def test_baremetal_unset_storage_interface(self):
        self._test_baremetal_unset_hw_interface('storage')

    def test_baremetal_unset_vendor_interface(self):
        self._test_baremetal_unset_hw_interface('vendor')

    def test_baremetal_unset_owner(self):
        arglist = ['node_uuid', '--owner']
        verifylist = [('nodes', ['node_uuid']), ('owner', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/owner', 'op': 'remove'}])

    def test_baremetal_unset_description(self):
        arglist = ['node_uuid', '--description']
        verifylist = [('nodes', ['node_uuid']), ('description', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/description', 'op': 'remove'}])

    def test_baremetal_unset_lessee(self):
        arglist = ['node_uuid', '--lessee']
        verifylist = [('nodes', ['node_uuid']), ('lessee', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/lessee', 'op': 'remove'}])

    def test_baremetal_unset_shard(self):
        arglist = ['node_uuid', '--shard']
        verifylist = [('nodes', ['node_uuid']), ('shard', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/shard', 'op': 'remove'}])

    def test_baremetal_unset_network_data(self):
        arglist = ['node_uuid', '--network-data']
        verifylist = [('nodes', ['node_uuid']), ('network_data', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.update.assert_called_once_with('node_uuid', [{'path': '/network_data', 'op': 'remove'}])