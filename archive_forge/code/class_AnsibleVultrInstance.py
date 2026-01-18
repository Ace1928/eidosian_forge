from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.common_instance import AnsibleVultrCommonInstance
from ..module_utils.vultr_v2 import vultr_argument_spec
class AnsibleVultrInstance(AnsibleVultrCommonInstance):

    def handle_power_status(self, resource, state, action, power_status, force=False, wait_for_state=True):
        if state == self.module.params['state'] and (resource['power_status'] != power_status or force):
            self.result['changed'] = True
            if not self.module.check_mode:
                resource = self.wait_for_state(resource=resource, key='server_status', states=['none', 'locked'], cmp='!=')
                self.api_query(path='%s/%s/%s' % (self.resource_path, resource[self.resource_key_id], action), method='POST')
                if wait_for_state:
                    resource = self.wait_for_state(resource=resource, key='power_status', states=[power_status])
        return resource

    def create_or_update(self):
        resource = super(AnsibleVultrInstance, self).create_or_update()
        if resource:
            resource = self.wait_for_state(resource=resource, key='server_status', states=['none', 'locked'], cmp='!=')
            resource = self.handle_power_status(resource=resource, state='stopped', action='halt', power_status='stopped')
            resource = self.handle_power_status(resource=resource, state='started', action='start', power_status='running')
            resource = self.handle_power_status(resource=resource, state='restarted', action='reboot', power_status='running', force=True)
            resource = self.handle_power_status(resource=resource, state='reinstalled', action='reinstall', power_status='running', force=True, wait_for_state=False)
        return resource

    def configure(self):
        super(AnsibleVultrInstance, self).configure()
        if self.module.params['state'] != 'absent':
            if self.module.params.get('firewall_group') is not None:
                self.module.params['firewall_group_id'] = self.get_firewall_group()['id']
            if self.module.params.get('backups') is not None:
                self.module.params['backups'] = 'enabled' if self.module.params['backups'] else 'disabled'

    def absent(self):
        resource = self.query()
        if resource and (not self.module.check_mode):
            resource = self.wait_for_state(resource=resource, key='server_status', states=['none', 'locked'], cmp='!=')
        return super(AnsibleVultrInstance, self).absent(resource=resource)