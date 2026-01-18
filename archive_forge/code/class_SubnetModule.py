from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class SubnetModule(OpenStackModule):
    ipv6_mode_choices = ['dhcpv6-stateful', 'dhcpv6-stateless', 'slaac']
    argument_spec = dict(name=dict(required=True), network=dict(aliases=['network_name']), cidr=dict(), description=dict(), ip_version=dict(type='int', default=4, choices=[4, 6]), is_dhcp_enabled=dict(type='bool', default=True, aliases=['enable_dhcp']), gateway_ip=dict(), disable_gateway_ip=dict(type='bool', default=False, aliases=['no_gateway_ip']), dns_nameservers=dict(type='list', elements='str'), allocation_pool_start=dict(), allocation_pool_end=dict(), host_routes=dict(type='list', elements='dict'), ipv6_ra_mode=dict(choices=ipv6_mode_choices), ipv6_address_mode=dict(choices=ipv6_mode_choices), subnet_pool=dict(), prefix_length=dict(), use_default_subnet_pool=dict(type='bool', aliases=['use_default_subnetpool']), extra_attrs=dict(type='dict', default=dict(), aliases=['extra_specs']), state=dict(default='present', choices=['absent', 'present']), project=dict())
    module_kwargs = dict(supports_check_mode=True, required_together=[['allocation_pool_end', 'allocation_pool_start']], required_if=[('state', 'present', ('network',)), ('state', 'present', ('cidr', 'use_default_subnet_pool', 'subnet_pool'), True)], mutually_exclusive=[('cidr', 'use_default_subnet_pool', 'subnet_pool')])
    attr_params = ('cidr', 'description', 'dns_nameservers', 'gateway_ip', 'host_routes', 'ip_version', 'ipv6_address_mode', 'ipv6_ra_mode', 'is_dhcp_enabled', 'name', 'prefix_length', 'use_default_subnet_pool')

    def _validate_update(self, subnet, update):
        """ Check for differences in non-updatable values """
        for attr in ('cidr', 'ip_version', 'ipv6_ra_mode', 'ipv6_address_mode', 'prefix_length', 'use_default_subnet_pool'):
            if attr in update and update[attr] != subnet[attr]:
                self.fail_json(msg='Cannot update {0} in existing subnet'.format(attr))

    def _system_state_change(self, subnet, network, project, subnet_pool):
        state = self.params['state']
        if state == 'absent':
            return subnet is not None
        if not subnet:
            return True
        params = self._build_params(network, project, subnet_pool)
        updates = self._build_updates(subnet, params)
        self._validate_update(subnet, updates)
        return bool(updates)

    def _build_pool(self):
        pool_start = self.params['allocation_pool_start']
        pool_end = self.params['allocation_pool_end']
        if pool_start:
            return [dict(start=pool_start, end=pool_end)]
        return None

    def _build_params(self, network, project, subnet_pool):
        params = {attr: self.params[attr] for attr in self.attr_params}
        params['network_id'] = network.id
        if project:
            params['project_id'] = project.id
        if subnet_pool:
            params['subnet_pool_id'] = subnet_pool.id
        params['allocation_pools'] = self._build_pool()
        params = self._add_extra_attrs(params)
        params = {k: v for k, v in params.items() if v is not None}
        return params

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

    def _add_extra_attrs(self, params):
        duplicates = set(self.params['extra_attrs']) & set(params)
        if duplicates:
            self.fail_json(msg='Duplicate key(s) {0} in extra_specs'.format(list(duplicates)))
        params.update(self.params['extra_attrs'])
        return params

    def run(self):
        state = self.params['state']
        network_name_or_id = self.params['network']
        project_name_or_id = self.params['project']
        subnet_pool_name_or_id = self.params['subnet_pool']
        subnet_name = self.params['name']
        gateway_ip = self.params['gateway_ip']
        disable_gateway_ip = self.params['disable_gateway_ip']
        if disable_gateway_ip and gateway_ip:
            self.fail_json(msg='no_gateway_ip is not allowed with gateway_ip')
        subnet_pool_filters = {}
        filters = {}
        project = None
        if project_name_or_id:
            project = self.conn.identity.find_project(project_name_or_id, ignore_missing=False)
            subnet_pool_filters['project_id'] = project.id
            filters['project_id'] = project.id
        network = None
        if network_name_or_id:
            network = self.conn.network.find_network(network_name_or_id, ignore_missing=False, **filters)
            filters['network_id'] = network.id
        subnet_pool = None
        if subnet_pool_name_or_id:
            subnet_pool = self.conn.network.find_subnet_pool(subnet_pool_name_or_id, ignore_missing=False, **subnet_pool_filters)
            filters['subnet_pool_id'] = subnet_pool.id
        subnet = self.conn.network.find_subnet(subnet_name, **filters)
        if self.ansible.check_mode:
            self.exit_json(changed=self._system_state_change(subnet, network, project, subnet_pool))
        changed = False
        if state == 'present':
            params = self._build_params(network, project, subnet_pool)
            if subnet is None:
                subnet = self.conn.network.create_subnet(**params)
                changed = True
            else:
                updates = self._build_updates(subnet, params)
                if updates:
                    self._validate_update(subnet, updates)
                    subnet = self.conn.network.update_subnet(subnet, **updates)
                    changed = True
            self.exit_json(changed=changed, subnet=subnet, id=subnet.id)
        elif state == 'absent' and subnet is not None:
            self.conn.network.delete_subnet(subnet)
            changed = True
        self.exit_json(changed=changed)