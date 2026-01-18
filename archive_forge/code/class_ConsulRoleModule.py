from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.consul import (
class ConsulRoleModule(_ConsulModule):
    api_endpoint = 'acl/role'
    result_key = 'role'
    unique_identifier = 'id'

    def endpoint_url(self, operation, identifier=None):
        if operation == OPERATION_READ:
            return [self.api_endpoint, 'name', self.params['name']]
        return super(ConsulRoleModule, self).endpoint_url(operation, identifier)