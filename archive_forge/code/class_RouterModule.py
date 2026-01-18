from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
from collections import defaultdict
class RouterModule(OpenStackModule):
    external_fixed_ips_spec = dict(type='list', elements='dict', options=dict(ip_address=dict(aliases=['ip']), subnet_id=dict(required=True, aliases=['subnet'])))
    argument_spec = dict(enable_snat=dict(type='bool'), external_fixed_ips=external_fixed_ips_spec, external_gateway_info=dict(type='dict', options=dict(network=dict(), enable_snat=dict(type='bool'), external_fixed_ips=external_fixed_ips_spec)), interfaces=dict(type='list', elements='raw'), is_admin_state_up=dict(type='bool', default=True, aliases=['admin_state_up']), name=dict(required=True), network=dict(), project=dict(), state=dict(default='present', choices=['absent', 'present']))
    module_kwargs = dict(mutually_exclusive=[('external_gateway_info', 'network'), ('external_gateway_info', 'external_fixed_ips'), ('external_gateway_info', 'enable_snat')], required_by={'external_fixed_ips': 'network', 'enable_snat': 'network'})

    def _needs_update(self, router, kwargs, external_fixed_ips, to_add, to_remove, missing_port_ids):
        """Decide if the given router needs an update."""
        if router['is_admin_state_up'] != self.params['is_admin_state_up']:
            return True
        cur_ext_gw_info = router['external_gateway_info']
        if 'external_gateway_info' in kwargs:
            if cur_ext_gw_info is None:
                return True
            update = kwargs['external_gateway_info']
            for attr in ('enable_snat', 'network_id'):
                if attr in update and cur_ext_gw_info[attr] != update[attr]:
                    return True
        cur_ext_gw_info = router['external_gateway_info']
        cur_ext_fips = (cur_ext_gw_info or {}).get('external_fixed_ips', [])
        cur_fip_map = defaultdict(set)
        for p in cur_ext_fips:
            if 'ip_address' in p:
                cur_fip_map[p['subnet_id']].add(p['ip_address'])
        req_fip_map = defaultdict(set)
        if external_fixed_ips is not None:
            for p in external_fixed_ips:
                if 'ip_address' in p:
                    req_fip_map[p['subnet_id']].add(p['ip_address'])
            for fip in external_fixed_ips:
                subnet = fip['subnet_id']
                ip = fip.get('ip_address', None)
                if subnet in cur_fip_map:
                    if ip is not None and ip not in cur_fip_map[subnet]:
                        return True
                else:
                    return True
            for fip in cur_ext_fips:
                subnet = fip['subnet_id']
                ip = fip['ip_address']
                if subnet in req_fip_map:
                    if ip not in req_fip_map[subnet]:
                        return True
                else:
                    return True
        if to_add or to_remove or missing_port_ids:
            return True
        return False

    def _build_kwargs(self, router, network, ext_fixed_ips):
        kwargs = {'is_admin_state_up': self.params['is_admin_state_up']}
        if not router:
            kwargs['name'] = self.params['name']
        external_gateway_info = {}
        if network:
            external_gateway_info['network_id'] = network.id
            if self.params['enable_snat'] is not None:
                external_gateway_info['enable_snat'] = self.params['enable_snat']
        if ext_fixed_ips:
            external_gateway_info['external_fixed_ips'] = ext_fixed_ips
        if external_gateway_info:
            kwargs['external_gateway_info'] = external_gateway_info
        if 'external_fixed_ips' not in external_gateway_info:
            curr_ext_gw_info = router['external_gateway_info'] if router else None
            curr_ext_fixed_ips = curr_ext_gw_info.get('external_fixed_ips', []) if curr_ext_gw_info else []
            if len(curr_ext_fixed_ips) > 1:
                external_gateway_info['external_fixed_ips'] = [curr_ext_fixed_ips[0]]
        return kwargs

    def _build_router_interface_config(self, filters):
        external_fixed_ips = None
        internal_ports_missing = []
        internal_ifaces = []
        ext_fixed_ips = None
        if self.params['external_gateway_info']:
            ext_fixed_ips = self.params['external_gateway_info'].get('external_fixed_ips')
        ext_fixed_ips = ext_fixed_ips or self.params['external_fixed_ips']
        if ext_fixed_ips:
            external_fixed_ips = []
            for iface in ext_fixed_ips:
                subnet = self.conn.network.find_subnet(iface['subnet_id'], ignore_missing=False, **filters)
                fip = dict(subnet_id=subnet.id)
                if 'ip_address' in iface:
                    fip['ip_address'] = iface['ip_address']
                external_fixed_ips.append(fip)
        if self.params['interfaces']:
            internal_ips = []
            for iface in self.params['interfaces']:
                if isinstance(iface, str):
                    subnet = self.conn.network.find_subnet(iface, ignore_missing=False, **filters)
                    internal_ifaces.append(dict(subnet_id=subnet.id))
                elif isinstance(iface, dict):
                    subnet = self.conn.network.find_subnet(iface['subnet'], ignore_missing=False, **filters)
                    if 'net' not in iface:
                        self.fail('Network name missing from interface definition')
                    net = self.conn.network.find_network(iface['net'], ignore_missing=False)
                    if 'portip' not in iface:
                        internal_ifaces.append(dict(subnet_id=subnet.id))
                    elif not iface['portip']:
                        self.fail(msg='put an ip in portip or remove itfrom list to assign default port to router')
                    else:
                        portip = iface['portip']
                        port_kwargs = {'network_id': net.id} if net is not None else {}
                        existing_ports = self.conn.network.ports(**port_kwargs)
                        for port in existing_ports:
                            for fip in port['fixed_ips']:
                                if fip['subnet_id'] != subnet.id or fip['ip_address'] != portip:
                                    continue
                                internal_ips.append(fip['ip_address'])
                                internal_ifaces.append(dict(port_id=port.id, subnet_id=subnet.id, ip_address=portip))
                        if portip not in internal_ips:
                            internal_ports_missing.append({'network_id': subnet.network_id, 'fixed_ips': [{'ip_address': portip, 'subnet_id': subnet.id}]})
        return {'external_fixed_ips': external_fixed_ips, 'internal_ports_missing': internal_ports_missing, 'internal_ifaces': internal_ifaces}

    def _update_ifaces(self, router, to_add, to_remove, missing_ports):
        for port in to_remove:
            self.conn.network.remove_interface_from_router(router, port_id=port.id)
        for port in missing_ports:
            p = self.conn.network.create_port(**port)
            if p:
                to_add.append(dict(port_id=p.id))
        for iface in to_add:
            self.conn.network.add_interface_to_router(router, **iface)

    def _get_external_gateway_network_name(self):
        network_name_or_id = self.params['network']
        if self.params['external_gateway_info']:
            network_name_or_id = self.params['external_gateway_info']['network']
        return network_name_or_id

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

    def run(self):
        state = self.params['state']
        name = self.params['name']
        network_name_or_id = self._get_external_gateway_network_name()
        project_name_or_id = self.params['project']
        if self.params['external_fixed_ips'] and (not network_name_or_id):
            self.fail(msg='network is required when supplying external_fixed_ips')
        query_filters = {}
        project = None
        project_id = None
        if project_name_or_id is not None:
            project = self.conn.identity.find_project(project_name_or_id, ignore_missing=False)
            project_id = project['id']
            query_filters['project_id'] = project_id
        router = self.conn.network.find_router(name, **query_filters)
        network = None
        if network_name_or_id:
            network = self.conn.network.find_network(network_name_or_id, ignore_missing=False, **query_filters)
        router_ifs_cfg = self._build_router_interface_config(query_filters)
        missing_internal_ports = router_ifs_cfg['internal_ports_missing']
        port_changes = self._get_port_changes(router, router_ifs_cfg)
        to_add = port_changes['to_add']
        to_remove = port_changes['to_remove']
        router_ifs_internal = port_changes['router_ifs_internal']
        external_fixed_ips = router_ifs_cfg['external_fixed_ips']
        if self.ansible.check_mode:
            if state == 'absent' and router:
                changed = True
            elif state == 'absent' and (not router):
                changed = False
            elif state == 'present' and (not router):
                changed = True
            else:
                kwargs = self._build_kwargs(router, network, external_fixed_ips)
                changed = self._needs_update(router, kwargs, external_fixed_ips, to_add, to_remove, missing_internal_ports)
            self.exit_json(changed=changed)
        if state == 'present':
            changed = False
            external_fixed_ips = router_ifs_cfg['external_fixed_ips']
            internal_ifaces = router_ifs_cfg['internal_ifaces']
            kwargs = self._build_kwargs(router, network, external_fixed_ips)
            if not router:
                changed = True
                if project_id:
                    kwargs['project_id'] = project_id
                router = self.conn.network.create_router(**kwargs)
                self._update_ifaces(router, internal_ifaces, [], missing_internal_ports)
            elif self._needs_update(router, kwargs, external_fixed_ips, to_add, to_remove, missing_internal_ports):
                changed = True
                router = self.conn.network.update_router(router, **kwargs)
                if to_add or to_remove or missing_internal_ports:
                    self._update_ifaces(router, to_add, to_remove, missing_internal_ports)
            self.exit_json(changed=changed, router=router.to_dict(computed=False))
        elif state == 'absent':
            if not router:
                self.exit_json(changed=False)
            else:
                for port in router_ifs_internal:
                    self.conn.network.remove_interface_from_router(router, port_id=port['id'])
                self.conn.network.delete_router(router)
                self.exit_json(changed=True)