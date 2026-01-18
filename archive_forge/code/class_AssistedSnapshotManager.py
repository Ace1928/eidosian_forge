from oslo_serialization import jsonutils
from novaclient import base
class AssistedSnapshotManager(base.Manager):
    resource_class = Snapshot

    def create(self, volume_id, create_info):
        body = {'snapshot': {'volume_id': volume_id, 'create_info': create_info}}
        return self._create('/os-assisted-volume-snapshots', body, 'snapshot')

    def delete(self, snapshot, delete_info):
        """
        Delete a specified assisted volume snapshot.

        :param snapshot: an assisted volume snapshot to delete
        :param delete_info: Information for snapshot deletion
        :returns: An instance of novaclient.base.TupleWithMeta
        """
        return self._delete('/os-assisted-volume-snapshots/%s?delete_info=%s' % (base.getid(snapshot), jsonutils.dumps(delete_info)))