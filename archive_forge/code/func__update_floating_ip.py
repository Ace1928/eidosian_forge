from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _update_floating_ip(self, load_balancer, update):
    floating_ip = None
    delete_floating_ip = update.get('delete_floating_ip')
    if delete_floating_ip:
        self.conn.network.delete_ip(delete_floating_ip.id)
    assign_floating_ip = update.get('assign_floating_ip')
    if assign_floating_ip:
        floating_ip_address = assign_floating_ip['floating_ip_address']
        floating_ip_network = assign_floating_ip['floating_ip_network']
        if floating_ip_network is not None:
            network = self.conn.network.find_network(floating_ip_network, ignore_missing=False)
        else:
            network = None
        if floating_ip_address is not None:
            kwargs = {'floating_network_id': network.id} if network is not None else {}
            ip = self.conn.network.find_ip(floating_ip_address, **kwargs)
        else:
            ip = None
        if ip:
            if ip['port_id'] is not None:
                self.fail_json(msg='Floating ip {0} is associated to another fixed ip address {1} already'.format(ip.floating_ip_address, ip.fixed_ip_address))
            floating_ip = self.conn.network.update_ip(ip.id, fixed_ip_address=load_balancer.vip_address, port_id=load_balancer.vip_port_id)
        elif floating_ip_address:
            kwargs = {'floating_network_id': network.id} if network is not None else {}
            floating_ip = self.conn.network.create_ip(fixed_ip_address=load_balancer.vip_address, floating_ip_address=floating_ip_address, port_id=load_balancer.vip_port_id, **kwargs)
        elif network:
            ips = [ip for ip in self.conn.network.ips(floating_network_id=network.id) if ip['port_id'] is None]
            if ips:
                ip = ips[0]
                floating_ip = self.conn.network.update_ip(ip.id, fixed_ip_address=load_balancer.vip_address, port_id=load_balancer.vip_port_id)
            else:
                floating_ip = self.conn.network.create_ip(fixed_ip_address=load_balancer.vip_address, floating_network_id=network.id, port_id=load_balancer.vip_port_id)
        else:
            ip = self.conn.network.find_available_ip()
            if ip:
                floating_ip = self.conn.network.update_ip(ip.id, fixed_ip_address=load_balancer.vip_address, port_id=load_balancer.vip_port_id)
            else:
                floating_ip = self.conn.network.create_ip(fixed_ip_address=load_balancer.vip_address, port_id=load_balancer.vip_port_id)
    return (load_balancer, floating_ip)