from unittest.mock import call
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group_snapshot
class TestConsistencyGroupSnapshotShow(TestConsistencyGroupSnapshot):
    _consistency_group_snapshot = volume_fakes.create_one_consistency_group_snapshot()
    columns = ('consistencygroup_id', 'created_at', 'description', 'id', 'name', 'status')
    data = (_consistency_group_snapshot.consistencygroup_id, _consistency_group_snapshot.created_at, _consistency_group_snapshot.description, _consistency_group_snapshot.id, _consistency_group_snapshot.name, _consistency_group_snapshot.status)

    def setUp(self):
        super(TestConsistencyGroupSnapshotShow, self).setUp()
        self.cgsnapshots_mock.get.return_value = self._consistency_group_snapshot
        self.cmd = consistency_group_snapshot.ShowConsistencyGroupSnapshot(self.app, None)

    def test_consistency_group_snapshot_show(self):
        arglist = [self._consistency_group_snapshot.id]
        verifylist = [('consistency_group_snapshot', self._consistency_group_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.cgsnapshots_mock.get.assert_called_once_with(self._consistency_group_snapshot.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)