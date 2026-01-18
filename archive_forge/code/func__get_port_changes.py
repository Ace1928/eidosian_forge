from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
from collections import defaultdict
def _get_port_changes(self, router, ifs_cfg):
    requested_subnet_ids = [iface['subnet_id'] for iface in ifs_cfg['internal_ifaces']]
    router_ifs_internal = []
    if router:
        router_ifs_internal = self.conn.list_router_interfaces(router, 'internal')
    existing_subnet_ips = {}
    for iface in router_ifs_internal:
        if 'fixed_ips' not in iface:
            continue
        for fip in iface['fixed_ips']:
            existing_subnet_ips[fip['subnet_id']] = (fip['ip_address'], iface)
    obsolete_subnet_ids = set(existing_subnet_ips.keys()) - set(requested_subnet_ids)
    internal_ifaces = ifs_cfg['internal_ifaces']
    to_add = []
    to_remove = []
    for iface in internal_ifaces:
        subnet_id = iface['subnet_id']
        if subnet_id not in existing_subnet_ips:
            iface.pop('ip_address', None)
            to_add.append(iface)
            continue
        ip, existing_port = existing_subnet_ips[subnet_id]
        if 'ip_address' in iface and ip != iface['ip_address']:
            to_remove.append(existing_port)
    for port in router_ifs_internal:
        if 'fixed_ips' not in port:
            continue
        if any((fip['subnet_id'] in obsolete_subnet_ids for fip in port['fixed_ips'])):
            to_remove.append(port)
    return dict(to_add=to_add, to_remove=to_remove, router_ifs_internal=router_ifs_internal)