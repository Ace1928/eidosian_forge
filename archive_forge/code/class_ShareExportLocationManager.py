from manilaclient import api_versions
from manilaclient import base
class ShareExportLocationManager(base.ManagerWithFind):
    """Manage :class:`ShareExportLocation` resources."""
    resource_class = ShareExportLocation

    @api_versions.wraps('2.9')
    def list(self, share, search_opts=None):
        """List all share export locations."""
        share_id = base.getid(share)
        return self._list('/shares/%s/export_locations' % share_id, 'export_locations')

    @api_versions.wraps('2.9')
    def get(self, share, export_location):
        """Get a share export location."""
        share_id = base.getid(share)
        export_location_id = base.getid(export_location)
        return self._get('/shares/%(share_id)s/export_locations/%(export_location_id)s' % {'share_id': share_id, 'export_location_id': export_location_id}, 'export_location')