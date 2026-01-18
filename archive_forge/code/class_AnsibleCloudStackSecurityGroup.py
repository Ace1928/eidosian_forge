from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
class AnsibleCloudStackSecurityGroup(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackSecurityGroup, self).__init__(module)
        self.security_group = None

    def get_security_group(self):
        if not self.security_group:
            args = {'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'securitygroupname': self.module.params.get('name')}
            sgs = self.query_api('listSecurityGroups', **args)
            if sgs:
                self.security_group = sgs['securitygroup'][0]
        return self.security_group

    def create_security_group(self):
        security_group = self.get_security_group()
        if not security_group:
            self.result['changed'] = True
            args = {'name': self.module.params.get('name'), 'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'description': self.module.params.get('description')}
            if not self.module.check_mode:
                res = self.query_api('createSecurityGroup', **args)
                security_group = res['securitygroup']
        return security_group

    def remove_security_group(self):
        security_group = self.get_security_group()
        if security_group:
            self.result['changed'] = True
            args = {'name': self.module.params.get('name'), 'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id')}
            if not self.module.check_mode:
                self.query_api('deleteSecurityGroup', **args)
        return security_group