from troveclient import base
class DatastoreVersions(base.ManagerWithFind):
    """Manage :class:`DatastoreVersion` resources."""
    resource_class = DatastoreVersion

    def __repr__(self):
        return '<DatastoreVersions Manager at %s>' % id(self)

    def list(self, datastore, limit=None, marker=None):
        """Get a list of all datastore versions.

        :rtype: list of :class:`DatastoreVersion`.
        """
        return self._paginated('/datastores/%s/versions' % datastore, 'versions', limit, marker)

    def get(self, datastore, datastore_version):
        """Get a specific datastore version.

        :rtype: :class:`DatastoreVersion`
        """
        return self._get('/datastores/%s/versions/%s' % (datastore, base.getid(datastore_version)), 'version')

    def get_by_uuid(self, datastore_version):
        """Get a specific datastore version.

        :rtype: :class:`DatastoreVersion`
        """
        return self._get('/datastores/versions/%s' % base.getid(datastore_version), 'version')

    def update(self, datastore, datastore_version, visibility):
        """Update a specific datastore version."""
        body = {'datastore_version': {}}
        if visibility is not None:
            body['datastore_version']['visibility'] = visibility
        url = '/mgmt/datastores/%s/versions/%s' % (datastore, datastore_version)
        return self._update(url, body=body)