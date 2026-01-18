from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _find_ips_by_nat_destination(self, server, nat_destination_name_or_id):
    if not server['addresses']:
        return None
    nat_destination = self.conn.network.find_network(nat_destination_name_or_id, ignore_missing=False)
    fips_with_nat_destination = [addr for addr in server['addresses'].get(nat_destination['name'], []) if addr['OS-EXT-IPS:type'] == 'floating']
    if not fips_with_nat_destination:
        return None
    return [self.conn.network.find_ip(fip['addr'], ignore_missing=False) for fip in fips_with_nat_destination]