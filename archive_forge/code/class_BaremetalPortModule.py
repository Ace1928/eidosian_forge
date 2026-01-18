from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
class BaremetalPortModule(OpenStackModule):
    argument_spec = dict(address=dict(), extra=dict(type='dict'), id=dict(aliases=['uuid']), is_pxe_enabled=dict(type='bool', aliases=['pxe_enabled']), local_link_connection=dict(type='dict'), node=dict(), physical_network=dict(), port_group=dict(aliases=['portgroup']), state=dict(default='present', choices=['present', 'absent']))
    module_kwargs = dict(required_one_of=[('id', 'address')], required_if=[('state', 'present', ('node', 'address'), False)])

    def run(self):
        port = self._find_port()
        state = self.params['state']
        if state == 'present':
            kwargs = {}
            id = self.params['id']
            if id:
                kwargs['id'] = id
            node_name_or_id = self.params['node']
            node = self.conn.baremetal.find_node(node_name_or_id, ignore_missing=False)
            kwargs['node_id'] = node['id']
            port_group_name_or_id = self.params['port_group']
            if port_group_name_or_id:
                port_group = self.conn.baremetal.find_port_group(port_group_name_or_id, ignore_missing=False)
                kwargs['port_group_id'] = port_group['id']
            for k in ['address', 'extra', 'is_pxe_enabled', 'local_link_connection', 'physical_network']:
                if self.params[k] is not None:
                    kwargs[k] = self.params[k]
            changed = True
            if not port:
                port = self.conn.baremetal.create_port(**kwargs)
            else:
                updates = dict(((k, v) for k, v in kwargs.items() if v != port[k]))
                if updates:
                    port = self.conn.baremetal.update_port(port['id'], **updates)
                else:
                    changed = False
            self.exit_json(changed=changed, port=port.to_dict(computed=False))
        if state == 'absent':
            if not port:
                self.exit_json(changed=False)
            port = self.conn.baremetal.delete_port(port['id'])
            self.exit_json(changed=True)

    def _find_port(self):
        id = self.params['id']
        if id:
            return self.conn.baremetal.get_port(id)
        address = self.params['address']
        if address:
            ports = list(self.conn.baremetal.ports(address=address, details=True))
            if len(ports) == 1:
                return ports[0]
            elif len(ports) > 1:
                raise ValueError('Multiple ports with address {address} found. A ID must be defined in order to identify a unique port.'.format(address=address))
            else:
                return None
        raise AssertionError('id or address must be specified')