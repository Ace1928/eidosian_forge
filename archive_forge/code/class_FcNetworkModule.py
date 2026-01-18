from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
class FcNetworkModule(OneViewModuleBase):
    MSG_CREATED = 'FC Network created successfully.'
    MSG_UPDATED = 'FC Network updated successfully.'
    MSG_DELETED = 'FC Network deleted successfully.'
    MSG_ALREADY_PRESENT = 'FC Network is already present.'
    MSG_ALREADY_ABSENT = 'FC Network is already absent.'
    RESOURCE_FACT_NAME = 'fc_network'

    def __init__(self):
        additional_arg_spec = dict(data=dict(required=True, type='dict'), state=dict(required=True, choices=['present', 'absent']))
        super(FcNetworkModule, self).__init__(additional_arg_spec=additional_arg_spec, validate_etag_support=True)
        self.resource_client = self.oneview_client.fc_networks

    def execute_module(self):
        resource = self.get_by_name(self.data['name'])
        if self.state == 'present':
            return self._present(resource)
        else:
            return self.resource_absent(resource)

    def _present(self, resource):
        scope_uris = self.data.pop('scopeUris', None)
        result = self.resource_present(resource, self.RESOURCE_FACT_NAME)
        if scope_uris is not None:
            result = self.resource_scopes_set(result, 'fc_network', scope_uris)
        return result