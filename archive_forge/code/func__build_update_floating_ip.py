from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _build_update_floating_ip(self, load_balancer):
    assign_floating_ip = self.params['assign_floating_ip']
    delete_floating_ip = self.params['delete_floating_ip']
    floating_ip_address = self.params['floating_ip_address']
    if floating_ip_address is not None and (not assign_floating_ip and (not delete_floating_ip)):
        self.fail_json(msg='assign_floating_ip or delete_floating_ip must be true when floating_ip_address is set')
    floating_ip_network = self.params['floating_ip_network']
    if floating_ip_network is not None and (not assign_floating_ip and (not delete_floating_ip)):
        self.fail_json(msg='assign_floating_ip or delete_floating_ip must be true when floating_ip_network is set')
    ips = list(self.conn.network.ips(port_id=load_balancer.vip_port_id, fixed_ip_address=load_balancer.vip_address))
    if len(ips) > 1:
        self.fail_json(msg='Only a single floating ip address per load-balancer is supported')
    if delete_floating_ip or not assign_floating_ip:
        if not ips:
            return (None, {})
        if len(ips) != 1:
            raise AssertionError('A single floating ip is expected')
        ip = ips[0]
        return (ip, {'delete_floating_ip': ip})
    if not ips:
        return (None, dict(assign_floating_ip=dict(floating_ip_address=floating_ip_address, floating_ip_network=floating_ip_network)))
    if len(ips) != 1:
        raise AssertionError('A single floating ip is expected')
    ip = ips[0]
    if floating_ip_network is not None:
        network = self.conn.network.find_network(floating_ip_network, ignore_missing=False)
        if ip.floating_network_id != network.id:
            return (ip, dict(assign_floating_ip=dict(floating_ip_address=floating_ip_address, floating_ip_network=floating_ip_network), delete_floating_ip=ip))
    if floating_ip_address is not None and floating_ip_address != ip.floating_ip_address:
        return (ip, dict(assign_floating_ip=dict(floating_ip_address=floating_ip_address, floating_ip_network=floating_ip_network), delete_floating_ip=ip))
    return (ip, {})