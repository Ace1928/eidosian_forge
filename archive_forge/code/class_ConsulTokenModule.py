from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.consul import (
class ConsulTokenModule(_ConsulModule):
    api_endpoint = 'acl/token'
    result_key = 'token'
    unique_identifier = 'accessor_id'
    create_only_fields = {'expiration_ttl'}

    def read_object(self):
        if not self.params.get(self.unique_identifier):
            return None
        return super(ConsulTokenModule, self).read_object()

    def needs_update(self, api_obj, module_obj):
        if 'SecretID' not in module_obj and 'SecretID' in api_obj:
            del api_obj['SecretID']
        normalize_link_obj(api_obj, module_obj, 'Roles')
        normalize_link_obj(api_obj, module_obj, 'Policies')
        if 'ExpirationTTL' in module_obj:
            del module_obj['ExpirationTTL']
        return super(ConsulTokenModule, self).needs_update(api_obj, module_obj)