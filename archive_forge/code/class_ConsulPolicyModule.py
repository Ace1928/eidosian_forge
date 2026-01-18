from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.consul import (
class ConsulPolicyModule(_ConsulModule):
    api_endpoint = 'acl/policy'
    result_key = 'policy'
    unique_identifier = 'id'

    def endpoint_url(self, operation, identifier=None):
        if operation == OPERATION_READ:
            return [self.api_endpoint, 'name', self.params['name']]
        return super(ConsulPolicyModule, self).endpoint_url(operation, identifier)