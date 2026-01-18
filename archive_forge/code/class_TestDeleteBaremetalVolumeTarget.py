import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestDeleteBaremetalVolumeTarget(TestBaremetalVolumeTarget):

    def setUp(self):
        super(TestDeleteBaremetalVolumeTarget, self).setUp()
        self.cmd = bm_vol_target.DeleteBaremetalVolumeTarget(self.app, None)

    def test_baremetal_volume_target_delete(self):
        arglist = [baremetal_fakes.baremetal_volume_target_uuid]
        verifylist = [('volume_targets', [baremetal_fakes.baremetal_volume_target_uuid])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.delete.assert_called_with(baremetal_fakes.baremetal_volume_target_uuid)

    def test_baremetal_volume_target_delete_multiple(self):
        fake_volume_target_uuid2 = 'vvv-tttttt-tttt'
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, fake_volume_target_uuid2]
        verifylist = [('volume_targets', [baremetal_fakes.baremetal_volume_target_uuid, fake_volume_target_uuid2])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_target.delete.assert_has_calls([mock.call(baremetal_fakes.baremetal_volume_target_uuid), mock.call(fake_volume_target_uuid2)])
        self.assertEqual(2, self.baremetal_mock.volume_target.delete.call_count)

    def test_baremetal_volume_target_delete_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_target_delete_error(self):
        arglist = [baremetal_fakes.baremetal_volume_target_uuid]
        verifylist = [('volume_targets', [baremetal_fakes.baremetal_volume_target_uuid])]
        self.baremetal_mock.volume_target.delete.side_effect = exc.NotFound()
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.ClientException, self.cmd.take_action, parsed_args)
        self.baremetal_mock.volume_target.delete.assert_called_with(baremetal_fakes.baremetal_volume_target_uuid)

    def test_baremetal_volume_target_delete_multiple_error(self):
        fake_volume_target_uuid2 = 'vvv-tttttt-tttt'
        arglist = [baremetal_fakes.baremetal_volume_target_uuid, fake_volume_target_uuid2]
        verifylist = [('volume_targets', [baremetal_fakes.baremetal_volume_target_uuid, fake_volume_target_uuid2])]
        self.baremetal_mock.volume_target.delete.side_effect = [None, exc.NotFound()]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.ClientException, self.cmd.take_action, parsed_args)
        self.baremetal_mock.volume_target.delete.assert_has_calls([mock.call(baremetal_fakes.baremetal_volume_target_uuid), mock.call(fake_volume_target_uuid2)])
        self.assertEqual(2, self.baremetal_mock.volume_target.delete.call_count)