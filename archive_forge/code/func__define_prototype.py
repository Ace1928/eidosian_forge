from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
def _define_prototype(self):
    filters = {}
    prototype = dict(((k, self.params[k]) for k in ['description', 'direction', 'remote_ip_prefix'] if self.params[k] is not None))
    project_name_or_id = self.params['project']
    if project_name_or_id is not None:
        project = self.conn.identity.find_project(project_name_or_id, ignore_missing=False)
        filters = {'project_id': project.id}
        prototype['project_id'] = project.id
    security_group_name_or_id = self.params['security_group']
    security_group = self.conn.network.find_security_group(security_group_name_or_id, ignore_missing=False, **filters)
    prototype['security_group_id'] = security_group.id
    remote_group = None
    remote_group_name_or_id = self.params['remote_group']
    if remote_group_name_or_id is not None:
        remote_group = self.conn.network.find_security_group(remote_group_name_or_id, ignore_missing=False)
        prototype['remote_group_id'] = remote_group.id
    ether_type = self.params['ether_type']
    if ether_type is not None:
        prototype['ether_type'] = ether_type
    protocol = self.params['protocol']
    if protocol is not None and protocol not in ['any', '0']:
        prototype['protocol'] = protocol
    port_range_max = self.params['port_range_max']
    port_range_min = self.params['port_range_min']
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