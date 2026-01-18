from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleResourceNotFound
def __present(self, resource):
    scope_uris = self.data.pop('scopeUris', None)
    self.__replace_name_by_uris(self.data)
    result = self.resource_present(resource, self.RESOURCE_FACT_NAME)
    if scope_uris is not None:
        result = self.resource_scopes_set(result, 'logical_interconnect_group', scope_uris)
    return result