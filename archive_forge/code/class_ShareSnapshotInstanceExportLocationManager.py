from manilaclient import api_versions
from manilaclient import base
class ShareSnapshotInstanceExportLocationManager(base.ManagerWithFind):
    """Manage :class:`ShareSnapshotInstanceExportLocation` resources."""
    resource_class = ShareSnapshotInstanceExportLocation

    @api_versions.wraps('2.32')
    def list(self, snapshot_instance=None, search_opts=None):
        return self._list('/snapshot-instances/%s/export-locations' % base.getid(snapshot_instance), 'share_snapshot_export_locations')

    @api_versions.wraps('2.32')
    def get(self, export_location, snapshot_instance=None):
        params = {'snapshot_instance_id': base.getid(snapshot_instance), 'export_location_id': base.getid(export_location)}
        return self._get('/snapshot-instances/%(snapshot_instance_id)s/export-locations/%(export_location_id)s' % params, 'share_snapshot_export_location')