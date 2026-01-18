from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
def _fetch_ports(self):
    name_or_id = self.params['name']
    if name_or_id:
        port = self.conn.baremetal.find_port(name_or_id)
        return [port] if port else []
    kwargs = {}
    address = self.params['address']
    if address:
        kwargs['address'] = address
    node_name_or_id = self.params['node']
    if node_name_or_id:
        node = self.conn.baremetal.find_node(node_name_or_id)
        if node:
            kwargs['node_uuid'] = node['id']
        else:
            return []
    return self.conn.baremetal.ports(details=True, **kwargs)