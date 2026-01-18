from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.vultr_v2 import AnsibleVultr, vultr_argument_spec
class AnsibleVultrUser(AnsibleVultr):

    def create(self):
        self.module.fail_on_missing_params(required_params=['password'])
        return super(AnsibleVultrUser, self).create()

    def update(self, resource):
        force = self.module.params.get('force')
        if force:
            self.resource_update_param_keys.append('password')
        return super(AnsibleVultrUser, self).update(resource=resource)