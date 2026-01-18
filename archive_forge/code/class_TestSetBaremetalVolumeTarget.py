import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestSetBaremetalVolumeTarget(TestBaremetalVolumeTarget):

    def setUp(self):
        super(TestSetBaremetalVolumeTarget, self).setUp()
        self.cmd = bm_vol_target.SetBaremetalVolumeTarget(self.app, None)

    def test_baremetal_volume_target_set_node_uuid(self):
        new_node_uuid = 'xxx-xxxxxx-zzzz'
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--node', new_node_uuid]
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('node_uuid', new_node_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/node_uuid', 'value': new_node_uuid, 'op': 'add'}])

    def test_baremetal_volume_target_set_volume_type(self):
        new_type = 'fibre_channel'
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--type', new_type]
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('volume_type', new_type)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/volume_type', 'value': new_type, 'op': 'add'}])

    def test_baremetal_volume_target_set_boot_index(self):
        new_boot_idx = '3'
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--boot-index', new_boot_idx]
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('boot_index', int(new_boot_idx))]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/boot_index', 'value': int(new_boot_idx), 'op': 'add'}])

    def test_baremetal_volume_target_set_negative_boot_index(self):
        new_boot_idx = '-3'
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--boot-index', new_boot_idx]
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('boot_index', int(new_boot_idx))]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)

    def test_baremetal_volume_target_set_invalid_boot_index(self):
        new_boot_idx = 'string'
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--boot-index', new_boot_idx]
        verifylist = None
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_target_set_volume_id(self):
        new_volume_id = 'new-volume-id'
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--volume-id', new_volume_id]
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('volume_id', new_volume_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/volume_id', 'value': new_volume_id, 'op': 'add'}])

    def test_baremetal_volume_target_set_volume_type_and_volume_id(self):
        new_volume_type = 'fibre_channel'
        new_volume_id = 'new-volume-id'
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--type', new_volume_type, '--volume-id', new_volume_id]
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('volume_type', new_volume_type), ('volume_id', new_volume_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/volume_type', 'value': new_volume_type, 'op': 'add'}, {'path': '/volume_id', 'value': new_volume_id, 'op': 'add'}])

    def test_baremetal_volume_target_set_extra(self):
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--extra', 'foo=bar']
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('extra', ['foo=bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/extra/foo', 'value': 'bar', 'op': 'add'}])

    def test_baremetal_volume_target_set_multiple_extras(self):
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--extra', 'key1=val1', '--extra', 'key2=val2']
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('extra', ['key1=val1', 'key2=val2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/extra/key1', 'value': 'val1', 'op': 'add'}, {'path': '/extra/key2', 'value': 'val2', 'op': 'add'}])

    def test_baremetal_volume_target_set_property(self):
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--property', 'foo=bar']
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('properties', ['foo=bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/properties/foo', 'value': 'bar', 'op': 'add'}])

    def test_baremetal_volume_target_set_multiple_properties(self):
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--property', 'key1=val1', '--property', 'key2=val2']
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid), ('properties', ['key1=val1', 'key2=val2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.update.assert_called_once_with(baremetal_fakes.baremetal_volume_target_uuid, [{'path': '/properties/key1', 'value': 'val1', 'op': 'add'}, {'path': '/properties/key2', 'value': 'val2', 'op': 'add'}])

    def test_baremetal_volume_target_set_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_target_set_no_property(self):
        arglist = [baremetal_fakes.baremetal_volume_target_uuid]
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.update.assert_not_called()