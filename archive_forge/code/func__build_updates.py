from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _build_updates(self, subnet, params):
    if 'dns_nameservers' in params:
        params['dns_nameservers'].sort()
        subnet['dns_nameservers'].sort()
    if 'host_routes' in params:
        params['host_routes'].sort(key=lambda r: sorted(r.items()))
        subnet['host_routes'].sort(key=lambda r: sorted(r.items()))
    updates = {k: params[k] for k in params if params[k] != subnet[k]}
    if self.params['disable_gateway_ip'] and subnet.gateway_ip:
        updates['gateway_ip'] = None
    return updates