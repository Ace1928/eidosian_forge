from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _get_replica_export_location(self, share_replica, export_location):
    """Get a share replica export location."""
    share_replica_id = base.getid(share_replica)
    export_location_id = base.getid(export_location)
    return self._get('/share-replicas/%(share_replica_id)s/export-locations/%(export_location_id)s' % {'share_replica_id': share_replica_id, 'export_location_id': export_location_id}, 'export_location')