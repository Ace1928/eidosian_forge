from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _find_ips_by_network_id_and_fixed_address(self, server, fixed_address=None, network_id=None):
    ips = [ip for ip in self.conn.network.ips() if ip['floating_ip_address'] in self._filter_ips(server)]
    matching_ips = []
    for ip in ips:
        if network_id and ip['floating_network_id'] != network_id:
            continue
        if not fixed_address:
            matching_ips.append(ip)
        if fixed_address and ip['fixed_ip_address'] == fixed_address:
            matching_ips.append(ip)
    return matching_ips