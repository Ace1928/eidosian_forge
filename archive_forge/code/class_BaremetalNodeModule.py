from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
class BaremetalNodeModule(OpenStackModule):
    argument_spec = dict(bios_interface=dict(), boot_interface=dict(), chassis_id=dict(aliases=['chassis_uuid']), console_interface=dict(), deploy_interface=dict(), driver=dict(), driver_info=dict(type='dict', required=True), id=dict(aliases=['uuid']), inspect_interface=dict(), management_interface=dict(), name=dict(), network_interface=dict(), nics=dict(type='list', required=True, elements='dict'), power_interface=dict(), properties=dict(type='dict', options=dict(cpu_arch=dict(), cpus=dict(), memory_mb=dict(aliases=['ram']), local_gb=dict(aliases=['disk_size']), capabilities=dict(), root_device=dict(type='dict'))), raid_interface=dict(), rescue_interface=dict(), resource_class=dict(), skip_update_of_masked_password=dict(type='bool', removed_in_version='3.0.0', removed_from_collection='openstack.cloud'), state=dict(default='present', choices=['present', 'absent']), storage_interface=dict(), timeout=dict(default=1800, type='int'), vendor_interface=dict())
    module_kwargs = dict(required_if=[('state', 'present', ('driver',))], required_one_of=[('id', 'name')], supports_check_mode=True)

    def run(self):
        name_or_id = self.params['id'] if self.params['id'] else self.params['name']
        node = self.conn.baremetal.find_node(name_or_id)
        state = self.params['state']
        if self.ansible.check_mode:
            self.exit_json(changed=self._will_change(state, node))
        if state == 'present' and (not node):
            node = self._create()
            self.exit_json(changed=True, node=node.to_dict(computed=False))
        elif state == 'present' and node:
            update = self._build_update(node)
            if update:
                node = self._update(node, update)
            self.exit_json(changed=bool(update), node=node.to_dict(computed=False))
        elif state == 'absent' and node:
            self._delete(node)
            self.exit_json(changed=True)
        elif state == 'absent' and (not node):
            self.exit_json(changed=False)

    def _build_update(self, node):
        update = {}
        node_attributes = dict(((k, self.params[k]) for k in ['bios_interface', 'boot_interface', 'chassis_id', 'console_interface', 'deploy_interface', 'driver', 'driver_info', 'inspect_interface', 'management_interface', 'name', 'network_interface', 'power_interface', 'raid_interface', 'rescue_interface', 'resource_class', 'storage_interface', 'vendor_interface'] if k in self.params and self.params[k] is not None and (self.params[k] != node[k])))
        properties = self.params['properties']
        if properties is not None:
            properties = dict(((k, v) for k, v in properties.items() if v is not None))
            if properties and properties != node['properties']:
                node_attributes['properties'] = properties
        if self.params['id'] is None and 'name' in node_attributes:
            self.fail_json(msg='The name of a node cannot be updated without specifying an id')
        if node_attributes:
            update['node_attributes'] = node_attributes
        return update

    def _create(self):
        kwargs = {}
        for k in ('bios_interface', 'boot_interface', 'chassis_id', 'console_interface', 'deploy_interface', 'driver', 'driver_info', 'id', 'inspect_interface', 'management_interface', 'name', 'network_interface', 'power_interface', 'raid_interface', 'rescue_interface', 'resource_class', 'storage_interface', 'vendor_interface'):
            if self.params[k] is not None:
                kwargs[k] = self.params[k]
        properties = self.params['properties']
        if properties is not None:
            properties = dict(((k, v) for k, v in properties.items() if v is not None))
            if properties:
                kwargs['properties'] = properties
        node = self.conn.register_machine(nics=self.params['nics'], wait=self.params['wait'], timeout=self.params['timeout'], **kwargs)
        self.exit_json(changed=True, node=node.to_dict(computed=False))

    def _delete(self, node):
        self.conn.unregister_machine(nics=self.params['nics'], uuid=node['id'])

    def _update(self, node, update):
        node_attributes = update.get('node_attributes')
        if node_attributes:
            node = self.conn.baremetal.update_node(node['id'], **node_attributes)
        return node

    def _will_change(self, state, node):
        if state == 'present' and (not node):
            return True
        elif state == 'present' and node:
            return bool(self._build_update(node))
        elif state == 'absent' and node:
            return True
        else:
            return False