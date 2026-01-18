from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackAffinityGroup(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackAffinityGroup, self).__init__(module)
        self.returns = {'type': 'affinity_type'}
        self.affinity_group = None

    def get_affinity_group(self):
        if not self.affinity_group:
            args = {'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'name': self.module.params.get('name')}
            affinity_groups = self.query_api('listAffinityGroups', **args)
            if affinity_groups:
                self.affinity_group = affinity_groups['affinitygroup'][0]
        return self.affinity_group

    def get_affinity_type(self):
        affinity_type = self.module.params.get('affinity_type')
        affinity_types = self.query_api('listAffinityGroupTypes')
        if affinity_types:
            if not affinity_type:
                return affinity_types['affinityGroupType'][0]['type']
            for a in affinity_types['affinityGroupType']:
                if a['type'] == affinity_type:
                    return a['type']
        self.module.fail_json(msg='affinity group type not found: %s' % affinity_type)

    def create_affinity_group(self):
        affinity_group = self.get_affinity_group()
        if not affinity_group:
            self.result['changed'] = True
            args = {'name': self.module.params.get('name'), 'type': self.get_affinity_type(), 'description': self.module.params.get('description'), 'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id')}
            if not self.module.check_mode:
                res = self.query_api('createAffinityGroup', **args)
                poll_async = self.module.params.get('poll_async')
                if res and poll_async:
                    affinity_group = self.poll_job(res, 'affinitygroup')
        return affinity_group

    def remove_affinity_group(self):
        affinity_group = self.get_affinity_group()
        if affinity_group:
            self.result['changed'] = True
            args = {'name': self.module.params.get('name'), 'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id')}
            if not self.module.check_mode:
                res = self.query_api('deleteAffinityGroup', **args)
                poll_async = self.module.params.get('poll_async')
                if res and poll_async:
                    self.poll_job(res, 'affinitygroup')
        return affinity_group