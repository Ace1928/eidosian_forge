from cinderclient.tests.unit.fixture_data import client
from cinderclient.tests.unit.fixture_data import snapshots
from cinderclient.tests.unit import utils
class SnapshotActionsTest(utils.FixturedTestCase):
    client_fixture_class = client.V3
    data_fixture_class = snapshots.Fixture

    def test_update_snapshot_status(self):
        snap = self.cs.volume_snapshots.get('1234')
        self._assert_request_id(snap)
        stat = {'status': 'available'}
        stats = self.cs.volume_snapshots.update_snapshot_status(snap, stat)
        self.assert_called('POST', '/snapshots/1234/action')
        self._assert_request_id(stats)

    def test_update_snapshot_status_with_progress(self):
        s = self.cs.volume_snapshots.get('1234')
        self._assert_request_id(s)
        stat = {'status': 'available', 'progress': '73%'}
        stats = self.cs.volume_snapshots.update_snapshot_status(s, stat)
        self.assert_called('POST', '/snapshots/1234/action')
        self._assert_request_id(stats)

    def test_list_snapshots_with_marker_limit(self):
        lst = self.cs.volume_snapshots.list(marker=1234, limit=2)
        self.assert_called('GET', '/snapshots/detail?limit=2&marker=1234')
        self._assert_request_id(lst)

    def test_list_snapshots_with_sort(self):
        lst = self.cs.volume_snapshots.list(sort='id')
        self.assert_called('GET', '/snapshots/detail?sort=id')
        self._assert_request_id(lst)

    def test_snapshot_unmanage(self):
        s = self.cs.volume_snapshots.get('1234')
        self._assert_request_id(s)
        snap = self.cs.volume_snapshots.unmanage(s)
        self.assert_called('POST', '/snapshots/1234/action', {'os-unmanage': None})
        self._assert_request_id(snap)