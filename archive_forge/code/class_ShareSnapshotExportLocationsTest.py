from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshot_export_locations
class ShareSnapshotExportLocationsTest(utils.TestCase):

    def test_list_snapshot(self):
        snapshot_id = '1234'
        cs.share_snapshot_export_locations.list(snapshot_id, search_opts=None)
        cs.assert_called('GET', '/snapshots/%s/export-locations' % snapshot_id)

    def test_get_snapshot(self):
        snapshot_id = '1234'
        el_id = 'fake_el_id'
        cs.share_snapshot_export_locations.get(el_id, snapshot_id)
        cs.assert_called('GET', '/snapshots/%(snapshot_id)s/export-locations/%(el_id)s' % {'snapshot_id': snapshot_id, 'el_id': el_id})