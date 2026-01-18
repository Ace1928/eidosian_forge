from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _generate_security_group_rule(params):
    prototype = dict(((k, params[k]) for k in ['description', 'direction', 'remote_ip_prefix'] if params[k] is not None))
    prototype['project_id'] = security_group.project_id
    prototype['security_group_id'] = security_group.id
    remote_group_name_or_id = params['remote_group']
    if remote_group_name_or_id is not None:
        if remote_group_name_or_id in security_group_cache:
            remote_group = security_group_cache[remote_group_name_or_id]
        else:
            remote_group = self.conn.network.find_security_group(remote_group_name_or_id, ignore_missing=False)
            security_group_cache[remote_group_name_or_id] = remote_group
        prototype['remote_group_id'] = remote_group.id
    ether_type = params['ether_type']
    if ether_type is not None:
        prototype['ether_type'] = ether_type
    protocol = params['protocol']
    if protocol is not None and protocol not in ['any', '0']:
        prototype['protocol'] = protocol
    port_range_max = params['port_range_max']
    port_range_min = params['port_range_min']
    if protocol in ['icmp', 'ipv6-icmp']:
        if port_range_max is not None and int(port_range_max) != -1:
            prototype['port_range_max'] = int(port_range_max)
        if port_range_min is not None and int(port_range_min) != -1:
            prototype['port_range_min'] = int(port_range_min)
    elif protocol in ['tcp', 'udp']:
        if port_range_max is not None and int(port_range_max) != -1:
            prototype['port_range_max'] = int(port_range_max)
        if port_range_min is not None and int(port_range_min) != -1:
            prototype['port_range_min'] = int(port_range_min)
    elif protocol in ['any', '0']:
        pass
    else:
        if port_range_max is not None:
            prototype['port_range_max'] = int(port_range_max)
        if port_range_min is not None:
            prototype['port_range_min'] = int(port_range_min)
    return prototype