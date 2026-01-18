from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as osc_lib_utils
from manilaclient.common.apiclient import exceptions as api_exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSnapshotInstanceSet(TestShareSnapshotInstance):

    def setUp(self):
        super(TestShareSnapshotInstanceSet, self).setUp()
        self.share_snapshot_instance = manila_fakes.FakeShareSnapshotIntances.create_one_snapshot_instance()
        self.snapshot_instance_status = 'available'
        self.cmd = osc_share_snapshot_instances.SetShareSnapshotInstance(self.app, None)

    def test_share_snapshot_instance_set_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_snapshot_instance_set_instance_not_found(self):
        arglist = [self.share_snapshot_instance.id, '--status', self.snapshot_instance_status]
        verifylist = [('snapshot_instance', self.share_snapshot_instance.id), ('status', self.snapshot_instance_status)]
        self.share_snapshot_instances_mock.reset_state.side_effect = api_exceptions.NotFound()
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_snapshot_instance_set(self):
        arglist = [self.share_snapshot_instance.id, '--status', self.snapshot_instance_status]
        verifylist = [('snapshot_instance', self.share_snapshot_instance.id), ('status', self.snapshot_instance_status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.share_snapshot_instances_mock.reset_state.assert_called_with(self.share_snapshot_instance.id, self.snapshot_instance_status)