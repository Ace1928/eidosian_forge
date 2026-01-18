from novaclient import base
from novaclient.i18n import _
class FlavorAccessManager(base.ManagerWithFind):
    """Manage :class:`FlavorAccess` resources."""
    resource_class = FlavorAccess

    def list(self, **kwargs):
        if kwargs.get('flavor'):
            return self._list('/flavors/%s/os-flavor-access' % base.getid(kwargs['flavor']), 'flavor_access')
        raise NotImplementedError(_('Unknown list options.'))

    def add_tenant_access(self, flavor, tenant):
        """Add a tenant to the given flavor access list."""
        info = {'tenant': tenant}
        return self._action('addTenantAccess', flavor, info)

    def remove_tenant_access(self, flavor, tenant):
        """Remove a tenant from the given flavor access list."""
        info = {'tenant': tenant}
        return self._action('removeTenantAccess', flavor, info)

    def _action(self, action, flavor, info, **kwargs):
        """Perform a flavor action."""
        body = {action: info}
        self.run_hooks('modify_body_for_action', body, **kwargs)
        url = '/flavors/%s/action' % base.getid(flavor)
        resp, body = self.api.client.post(url, body=body)
        items = [self.resource_class(self, res) for res in body['flavor_access']]
        return base.ListWithMeta(items, resp)