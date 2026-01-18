from keystoneclient import base
class EndpointManager(base.ManagerWithFind):
    """Manager class for manipulating Keystone endpoints."""
    resource_class = Endpoint

    def list(self):
        """List all available endpoints."""
        return self._list('/endpoints', 'endpoints')

    def create(self, region, service_id, publicurl, adminurl=None, internalurl=None):
        """Create a new endpoint."""
        body = {'endpoint': {'region': region, 'service_id': service_id, 'publicurl': publicurl, 'adminurl': adminurl, 'internalurl': internalurl}}
        return self._post('/endpoints', body, 'endpoint')

    def delete(self, id):
        """Delete an endpoint."""
        return self._delete('/endpoints/%s' % id)