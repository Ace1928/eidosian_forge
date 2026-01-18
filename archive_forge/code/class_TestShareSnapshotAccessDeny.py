from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSnapshotAccessDeny(TestShareSnapshot):

    def setUp(self):
        super(TestShareSnapshotAccessDeny, self).setUp()
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot()
        self.snapshots_mock.get.return_value = self.share_snapshot
        self.access_rule = manila_fakes.FakeSnapshotAccessRule.create_one_access_rule()
        self.cmd = osc_share_snapshots.ShareSnapshotAccessDeny(self.app, None)

    def test_share_snapshot_access_deny(self):
        arglist = [self.share_snapshot.id, self.access_rule.id]
        verifylist = [('snapshot', self.share_snapshot.id), ('id', [self.access_rule.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.snapshots_mock.deny.assert_called_with(snapshot=self.share_snapshot, id=self.access_rule.id)
        self.assertIsNone(result)

    def test_share_snapshot_access_deny_multiple(self):
        access_rules = manila_fakes.FakeSnapshotAccessRule.create_access_rules(count=2)
        arglist = [self.share_snapshot.id, access_rules[0].id, access_rules[1].id]
        verifylist = [('snapshot', self.share_snapshot.id), ('id', [access_rules[0].id, access_rules[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertEqual(self.snapshots_mock.deny.call_count, len(access_rules))
        self.assertIsNone(result)

    def test_share_snapshot_access_deny_exception(self):
        arglist = [self.share_snapshot.id, self.access_rule.id]
        verifylist = [('snapshot', self.share_snapshot.id), ('id', [self.access_rule.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.snapshots_mock.deny.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)