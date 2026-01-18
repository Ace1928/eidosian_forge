from unittest.mock import call
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group_snapshot
class TestConsistencyGroupSnapshotDelete(TestConsistencyGroupSnapshot):
    consistency_group_snapshots = volume_fakes.create_consistency_group_snapshots(count=2)

    def setUp(self):
        super(TestConsistencyGroupSnapshotDelete, self).setUp()
        self.cgsnapshots_mock.get = volume_fakes.get_consistency_group_snapshots(self.consistency_group_snapshots)
        self.cgsnapshots_mock.delete.return_value = None
        self.cmd = consistency_group_snapshot.DeleteConsistencyGroupSnapshot(self.app, None)

    def test_consistency_group_snapshot_delete(self):
        arglist = [self.consistency_group_snapshots[0].id]
        verifylist = [('consistency_group_snapshot', [self.consistency_group_snapshots[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.cgsnapshots_mock.delete.assert_called_once_with(self.consistency_group_snapshots[0].id)
        self.assertIsNone(result)

    def test_multiple_consistency_group_snapshots_delete(self):
        arglist = []
        for c in self.consistency_group_snapshots:
            arglist.append(c.id)
        verifylist = [('consistency_group_snapshot', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for c in self.consistency_group_snapshots:
            calls.append(call(c.id))
        self.cgsnapshots_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)