from manilaclient import api_versions
from manilaclient import base
class ShareSnapshotExportLocationManager(base.ManagerWithFind):
    """Manage :class:`ShareSnapshotExportLocation` resources."""
    resource_class = ShareSnapshotExportLocation

    @api_versions.wraps('2.32')
    def list(self, snapshot=None, search_opts=None):
        return self._list('/snapshots/%s/export-locations' % base.getid(snapshot), 'share_snapshot_export_locations')

    @api_versions.wraps('2.32')
    def get(self, export_location, snapshot=None):
        params = {'snapshot_id': base.getid(snapshot), 'export_location_id': base.getid(export_location)}
        return self._get('/snapshots/%(snapshot_id)s/export-locations/%(export_location_id)s' % params, 'share_snapshot_export_location')