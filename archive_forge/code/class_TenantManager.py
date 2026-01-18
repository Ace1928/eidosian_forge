from keystoneauth1 import plugin
import urllib.parse
from keystoneclient import base
from keystoneclient import exceptions
class TenantManager(base.ManagerWithFind):
    """Manager class for manipulating Keystone tenants."""
    resource_class = Tenant

    def __init__(self, client, role_manager, user_manager):
        super(TenantManager, self).__init__(client)
        self.role_manager = role_manager
        self.user_manager = user_manager

    def get(self, tenant_id):
        return self._get('/tenants/%s' % tenant_id, 'tenant')

    def create(self, tenant_name, description=None, enabled=True, **kwargs):
        """Create a new tenant."""
        params = {'tenant': {'name': tenant_name, 'description': description, 'enabled': enabled}}
        for k, v in kwargs.items():
            if k not in params['tenant']:
                params['tenant'][k] = v
        return self._post('/tenants', params, 'tenant')

    def list(self, limit=None, marker=None):
        """Get a list of tenants.

        :param integer limit: maximum number to return. (optional)
        :param string marker: use when specifying a limit and making
                              multiple calls for querying. (optional)

        :rtype: list of :class:`Tenant`

        """
        params = {}
        if limit:
            params['limit'] = limit
        if marker:
            params['marker'] = marker
        query = ''
        if params:
            query = '?' + urllib.parse.urlencode(params)
        try:
            tenant_list = self._list('/tenants%s' % query, 'tenants')
        except exceptions.EndpointNotFound:
            endpoint_filter = {'interface': plugin.AUTH_INTERFACE}
            tenant_list = self._list('/tenants%s' % query, 'tenants', endpoint_filter=endpoint_filter)
        return tenant_list

    def update(self, tenant_id, tenant_name=None, description=None, enabled=None, **kwargs):
        """Update a tenant with a new name and description."""
        body = {'tenant': {'id': tenant_id}}
        if tenant_name is not None:
            body['tenant']['name'] = tenant_name
        if enabled is not None:
            body['tenant']['enabled'] = enabled
        if description is not None:
            body['tenant']['description'] = description
        for k, v in kwargs.items():
            if k not in body['tenant']:
                body['tenant'][k] = v
        return self._post('/tenants/%s' % tenant_id, body, 'tenant')

    def delete(self, tenant):
        """Delete a tenant."""
        return self._delete('/tenants/%s' % base.getid(tenant))

    def list_users(self, tenant):
        """List users for a tenant."""
        return self.user_manager.list(base.getid(tenant))

    def add_user(self, tenant, user, role):
        """Add a user to a tenant with the given role."""
        return self.role_manager.add_user_role(base.getid(user), base.getid(role), base.getid(tenant))

    def remove_user(self, tenant, user, role):
        """Remove the specified role from the user on the tenant."""
        return self.role_manager.remove_user_role(base.getid(user), base.getid(role), base.getid(tenant))