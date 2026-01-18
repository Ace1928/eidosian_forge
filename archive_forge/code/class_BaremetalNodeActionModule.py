from ansible_collections.openstack.cloud.plugins.module_utils.openstack import (
class BaremetalNodeActionModule(OpenStackModule):
    argument_spec = dict(config_drive=dict(type='raw'), deploy=dict(type='bool', default=True), instance_info=dict(type='dict'), maintenance=dict(type='bool'), maintenance_reason=dict(), name=dict(required=True, aliases=['id', 'uuid']), power=dict(default='present', choices=['present', 'absent', 'maintenance', 'on', 'off']), state=dict(default='present', choices=['present', 'absent', 'maintenance', 'on', 'off']), timeout=dict(type='int', default=1800))
    module_kwargs = dict(required_if=[('state', 'present', ('instance_info',))])

    def run(self):
        config_drive = self.params['config_drive']
        if config_drive and (not isinstance(config_drive, (str, dict))):
            self.fail_json(msg='config_drive must be of type str or dict, not {0}'.format(type(config_drive)))
        if self.params['state'] == 'maintenance':
            if self.params['maintenance'] is False:
                self.fail_json(msg='state=maintenance contradicts with maintenance=false')
            self.params['maintenance'] = True
        name_or_id = self.params['name']
        node = self.conn.baremetal.find_node(name_or_id, ignore_missing=False)
        if node['provision_state'] in ['cleaning', 'deleting', 'wait call-back']:
            self.fail_json(msg='Node is in {0} state, cannot act upon the request as the node is in a transition state'.format(node['provision_state']))
        changed = False
        if self.params['maintenance']:
            maintenance_reason = self.params['maintenance_reason']
            if not node['maintenance'] or node['maintenance_reason'] != maintenance_reason:
                self.conn.baremetal.set_node_maintenance(node['id'], reason=maintenance_reason)
                self.exit_json(changed=True)
        elif node['maintenance']:
            self.conn.baremetal.unset_node_maintenance(node['id'])
            if node['provision_state'] in 'active':
                self.exit_json(changed=True)
            changed = True
            node = self.conn.baremetal.get_node(node['id'])
        if node['power_state'] == 'power on':
            if self.params['power'] in ['absent', 'off']:
                self.conn.baremetal.set_node_power_state(node['id'], 'power off')
                self.exit_json(changed=True)
        elif node['power_state'] == 'power off':
            if self.params['power'] not in ['absent', 'off'] or self.params['state'] not in ['absent', 'off']:
                if self.params['power'] == 'present' and (not self.params['deploy']):
                    self.conn.baremetal.set_node_power_state(node['id'], 'power on')
                    changed = True
                    node = self.conn.baremetal.get_node(node['id'])
        else:
            self.fail_json(msg='Node has unknown power state {0}'.format(node['power_state']))
        if self.params['state'] in ['present', 'on']:
            if not self.params['deploy']:
                self.exit_json(changed=changed)
            if 'active' in node['provision_state']:
                self.exit_json(changed=changed)
            self.conn.baremetal.update_node(node['id'], instance_info=self.params['instance_info'])
            self.conn.baremetal.validate_node(node['id'])
            self.conn.baremetal.set_node_provision_state(node['id'], target='active', config_drive=self.params['config_drive'], wait=self.params['wait'], timeout=self.params['timeout'])
            self.exit_json(changed=True)
        elif node['provision_state'] not in 'deleted':
            self.conn.baremetal.update_node(node['id'], instance_info={})
            self.conn.baremetal.set_node_provision_state(node['id'], target='deleted', wait=self.params['wait'], timeout=self.params['timeout'])
            self.exit_json(changed=True)
        else:
            self.exit_json(changed=changed)