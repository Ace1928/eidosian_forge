from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
from collections import defaultdict
def _update_ifaces(self, router, to_add, to_remove, missing_ports):
    for port in to_remove:
        self.conn.network.remove_interface_from_router(router, port_id=port.id)
    for port in missing_ports:
        p = self.conn.network.create_port(**port)
        if p:
            to_add.append(dict(port_id=p.id))
    for iface in to_add:
        self.conn.network.add_interface_to_router(router, **iface)