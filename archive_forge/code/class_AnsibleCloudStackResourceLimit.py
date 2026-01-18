from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackResourceLimit(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackResourceLimit, self).__init__(module)
        self.returns = {'max': 'limit'}

    def get_resource_type(self):
        resource_type = self.module.params.get('resource_type')
        return RESOURCE_TYPES.get(resource_type)

    def get_resource_limit(self):
        args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'resourcetype': self.get_resource_type()}
        resource_limit = self.query_api('listResourceLimits', **args)
        if resource_limit:
            if 'limit' in resource_limit['resourcelimit'][0]:
                resource_limit['resourcelimit'][0]['limit'] = int(resource_limit['resourcelimit'][0])
            return resource_limit['resourcelimit'][0]
        self.module.fail_json(msg="Resource limit type '%s' not found." % self.module.params.get('resource_type'))

    def update_resource_limit(self):
        resource_limit = self.get_resource_limit()
        args = {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'resourcetype': self.get_resource_type(), 'max': self.module.params.get('limit', -1)}
        if self.has_changed(args, resource_limit):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateResourceLimit', **args)
                resource_limit = res['resourcelimit']
        return resource_limit

    def get_result(self, resource):
        self.result = super(AnsibleCloudStackResourceLimit, self).get_result(resource)
        self.result['resource_type'] = self.module.params.get('resource_type')
        return self.result