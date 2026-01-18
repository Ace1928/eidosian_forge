from troveclient import base
class DatastoreVersionMembers(base.ManagerWithFind):
    """Manage :class:`DatastoreVersionMember` resources."""
    resource_class = DatastoreVersionMember

    def __repr__(self):
        return '<DatastoreVersionMembers Manager at %s>' % id(self)

    def add(self, datastore, datastore_version, tenant):
        """Add a member to a datastore version."""
        body = {'member': tenant}
        return self._create('/mgmt/datastores/%s/versions/%s/members' % (datastore, datastore_version), body, 'datastore_version_member')

    def delete(self, datastore, datastore_version, member_id):
        """Delete a member from a datastore version."""
        return self._delete('/mgmt/datastores/%s/versions/%s/members/%s' % (datastore, datastore_version, member_id))

    def list(self, datastore, datastore_version, limit=None, marker=None):
        """List members of datastore version."""
        return self._list('/mgmt/datastores/%s/versions/%s/members' % (datastore, datastore_version), 'datastore_version_members', limit, marker)

    def get(self, datastore, datastore_version, member_id):
        """Get a datastore version member."""
        return self._get('/mgmt/datastores/%s/versions/%s/members/%s' % (datastore, datastore_version, member_id), 'datastore_version_member')

    def get_by_tenant(self, datastore, tenant, limit=None, marker=None):
        """List members by tenant id."""
        return self._list('/mgmt/datastores/%s/versions/members/%s' % (datastore, tenant), 'datastore_version_members', limit, marker)