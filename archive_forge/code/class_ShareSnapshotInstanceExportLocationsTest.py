from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshot_instance_export_locations
class ShareSnapshotInstanceExportLocationsTest(utils.TestCase):

    def test_list_snapshot_instance(self):
        snapshot_instance_id = '1234'
        cs.share_snapshot_instance_export_locations.list(snapshot_instance_id, search_opts=None)
        cs.assert_called('GET', '/snapshot-instances/%s/export-locations' % snapshot_instance_id)

    def test_get_snapshot_instance(self):
        snapshot_instance_id = '1234'
        el_id = 'fake_el_id'
        cs.share_snapshot_instance_export_locations.get(el_id, snapshot_instance_id)
        cs.assert_called('GET', '/snapshot-instances/%(snapshot_id)s/export-locations/%(el_id)s' % {'snapshot_id': snapshot_instance_id, 'el_id': el_id})