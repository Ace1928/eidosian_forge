import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestCreateBaremetalVolumeTarget(TestBaremetalVolumeTarget):

    def setUp(self):
        super(TestCreateBaremetalVolumeTarget, self).setUp()
        self.baremetal_mock.volume_target.create.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.VOLUME_TARGET), loaded=True)
        self.cmd = bm_vol_target.CreateBaremetalVolumeTarget(self.app, None)

    def test_baremetal_volume_target_create(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', baremetal_fakes.baremetal_volume_target_volume_type, '--boot-index', baremetal_fakes.baremetal_volume_target_boot_index, '--volume-id', baremetal_fakes.baremetal_volume_target_volume_id, '--uuid', baremetal_fakes.baremetal_volume_target_uuid]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('volume_type', baremetal_fakes.baremetal_volume_target_volume_type), ('boot_index', baremetal_fakes.baremetal_volume_target_boot_index), ('volume_id', baremetal_fakes.baremetal_volume_target_volume_id), ('uuid', baremetal_fakes.baremetal_volume_target_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'node_uuid': baremetal_fakes.baremetal_uuid, 'volume_type': baremetal_fakes.baremetal_volume_target_volume_type, 'boot_index': baremetal_fakes.baremetal_volume_target_boot_index, 'volume_id': baremetal_fakes.baremetal_volume_target_volume_id, 'uuid': baremetal_fakes.baremetal_volume_target_uuid}
        self.baremetal_mock.volume_target.create.assert_called_once_with(**args)

    def test_baremetal_volume_target_create_without_uuid(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', baremetal_fakes.baremetal_volume_target_volume_type, '--boot-index', baremetal_fakes.baremetal_volume_target_boot_index, '--volume-id', baremetal_fakes.baremetal_volume_target_volume_id]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('volume_type', baremetal_fakes.baremetal_volume_target_volume_type), ('boot_index', baremetal_fakes.baremetal_volume_target_boot_index), ('volume_id', baremetal_fakes.baremetal_volume_target_volume_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'node_uuid': baremetal_fakes.baremetal_uuid, 'volume_type': baremetal_fakes.baremetal_volume_target_volume_type, 'boot_index': baremetal_fakes.baremetal_volume_target_boot_index, 'volume_id': baremetal_fakes.baremetal_volume_target_volume_id}
        self.baremetal_mock.volume_target.create.assert_called_once_with(**args)

    def test_baremetal_volume_target_create_extras(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', baremetal_fakes.baremetal_volume_target_volume_type, '--boot-index', baremetal_fakes.baremetal_volume_target_boot_index, '--volume-id', baremetal_fakes.baremetal_volume_target_volume_id, '--extra', 'key1=value1', '--extra', 'key2=value2']
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('volume_type', baremetal_fakes.baremetal_volume_target_volume_type), ('boot_index', baremetal_fakes.baremetal_volume_target_boot_index), ('volume_id', baremetal_fakes.baremetal_volume_target_volume_id), ('extra', ['key1=value1', 'key2=value2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'node_uuid': baremetal_fakes.baremetal_uuid, 'volume_type': baremetal_fakes.baremetal_volume_target_volume_type, 'boot_index': baremetal_fakes.baremetal_volume_target_boot_index, 'volume_id': baremetal_fakes.baremetal_volume_target_volume_id, 'extra': baremetal_fakes.baremetal_volume_target_extra}
        self.baremetal_mock.volume_target.create.assert_called_once_with(**args)

    def _test_baremetal_volume_target_missing_param(self, missing):
        argdict = {'--node': baremetal_fakes.baremetal_uuid, '--type': baremetal_fakes.baremetal_volume_target_volume_type, '--boot-index': baremetal_fakes.baremetal_volume_target_boot_index, '--volume-id': baremetal_fakes.baremetal_volume_target_volume_id, '--uuid': baremetal_fakes.baremetal_volume_target_uuid}
        arglist = []
        for k, v in argdict.items():
            if k not in missing:
                arglist += [k, v]
        verifylist = None
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_target_create_missing_node(self):
        self._test_baremetal_volume_target_missing_param(['--node'])

    def test_baremetal_volume_target_create_missing_type(self):
        self._test_baremetal_volume_target_missing_param(['--type'])

    def test_baremetal_volume_target_create_missing_boot_index(self):
        self._test_baremetal_volume_target_missing_param(['--boot-index'])

    def test_baremetal_volume_target_create_missing_volume_id(self):
        self._test_baremetal_volume_target_missing_param(['--volume-id'])

    def test_baremetal_volume_target_create_invalid_boot_index(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', baremetal_fakes.baremetal_volume_target_volume_type, '--boot-index', 'string', '--volume-id', baremetal_fakes.baremetal_volume_target_volume_id]
        verifylist = None
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_target_create_negative_boot_index(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', baremetal_fakes.baremetal_volume_target_volume_type, '--boot-index', '-1', '--volume-id', baremetal_fakes.baremetal_volume_target_volume_id]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('volume_type', baremetal_fakes.baremetal_volume_target_volume_type), ('boot_index', -1), ('volume_id', baremetal_fakes.baremetal_volume_target_volume_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)