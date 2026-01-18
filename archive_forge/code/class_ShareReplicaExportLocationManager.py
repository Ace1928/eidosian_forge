from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
class ShareReplicaExportLocationManager(base.ManagerWithFind):
    """Manage :class:`ShareInstanceExportLocation` resources."""
    resource_class = ShareReplicaExportLocation

    @api_versions.wraps('2.47', constants.REPLICA_PRE_GRADUATION_VERSION)
    @api_versions.experimental_api
    def list(self, share_replica, search_opts=None):
        """List all share replica export locations."""
        share_replica_id = base.getid(share_replica)
        return self._list('/share-replicas/%s/export-locations' % share_replica_id, 'export_locations')

    @api_versions.wraps(constants.REPLICA_GRADUATION_VERSION)
    def list(self, share_replica, search_opts=None):
        """List all share replica export locations."""
        share_replica_id = base.getid(share_replica)
        return self._list('/share-replicas/%s/export-locations' % share_replica_id, 'export_locations')

    @api_versions.wraps('2.47', constants.REPLICA_PRE_GRADUATION_VERSION)
    @api_versions.experimental_api
    def get(self, share_replica, export_location):
        return self._get_replica_export_location(share_replica, export_location)

    @api_versions.wraps(constants.REPLICA_GRADUATION_VERSION)
    def get(self, share_replica, export_location):
        return self._get_replica_export_location(share_replica, export_location)

    def _get_replica_export_location(self, share_replica, export_location):
        """Get a share replica export location."""
        share_replica_id = base.getid(share_replica)
        export_location_id = base.getid(export_location)
        return self._get('/share-replicas/%(share_replica_id)s/export-locations/%(export_location_id)s' % {'share_replica_id': share_replica_id, 'export_location_id': export_location_id}, 'export_location')