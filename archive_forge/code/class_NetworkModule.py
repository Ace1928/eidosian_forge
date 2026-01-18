from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
class NetworkModule(OpenStackModule):
    argument_spec = dict(name=dict(required=True), shared=dict(type='bool'), admin_state_up=dict(type='bool'), external=dict(type='bool'), provider_physical_network=dict(), provider_network_type=dict(), provider_segmentation_id=dict(type='int'), state=dict(default='present', choices=['absent', 'present']), project=dict(), port_security_enabled=dict(type='bool'), mtu=dict(type='int', aliases=['mtu_size']), dns_domain=dict())

    def run(self):
        state = self.params['state']
        name = self.params['name']
        shared = self.params['shared']
        admin_state_up = self.params['admin_state_up']
        external = self.params['external']
        provider_physical_network = self.params['provider_physical_network']
        provider_network_type = self.params['provider_network_type']
        provider_segmentation_id = self.params['provider_segmentation_id']
        project = self.params['project']
        kwargs = {}
        for arg in ('port_security_enabled', 'mtu', 'dns_domain'):
            if self.params[arg] is not None:
                kwargs[arg] = self.params[arg]
        if project is not None:
            proj = self.conn.identity.find_project(project, ignore_missing=False)
            project_id = proj['id']
            net_kwargs = {'project_id': project_id}
        else:
            project_id = None
            net_kwargs = {}
        net = self.conn.network.find_network(name, **net_kwargs)
        if state == 'present':
            if provider_physical_network:
                kwargs['provider_physical_network'] = provider_physical_network
            if provider_network_type:
                kwargs['provider_network_type'] = provider_network_type
            if provider_segmentation_id:
                kwargs['provider_segmentation_id'] = provider_segmentation_id
            if project_id is not None:
                kwargs['project_id'] = project_id
            if shared is not None:
                kwargs['shared'] = shared
            if admin_state_up is not None:
                kwargs['admin_state_up'] = admin_state_up
            if external is not None:
                kwargs['is_router_external'] = external
            if not net:
                net = self.conn.network.create_network(name=name, **kwargs)
                changed = True
            else:
                changed = False
                update_kwargs = {}
                non_updatables = ['provider_network_type', 'provider_physical_network']
                for arg in non_updatables:
                    if arg in kwargs and kwargs[arg] != net[arg]:
                        self.fail_json(msg='The following parameters cannot be updated: %s. You will need to use state: absent and recreate.' % ', '.join(non_updatables))
                for arg in ['shared', 'admin_state_up', 'is_router_external', 'mtu', 'port_security_enabled', 'dns_domain', 'provider_segmentation_id']:
                    if arg in kwargs and kwargs[arg] is not None and (kwargs[arg] != net[arg]):
                        update_kwargs[arg] = kwargs[arg]
                if update_kwargs:
                    net = self.conn.network.update_network(net.id, **update_kwargs)
                    changed = True
            net = net.to_dict(computed=False)
            self.exit(changed=changed, network=net, id=net['id'])
        elif state == 'absent':
            if not net:
                self.exit(changed=False)
            else:
                self.conn.network.delete_network(net['id'])
                self.exit(changed=True)