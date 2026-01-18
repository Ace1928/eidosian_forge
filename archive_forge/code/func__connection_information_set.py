from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleValueError
def _connection_information_set(self, resource):
    if not resource:
        return self._present(resource)
    else:
        merged_data = resource.copy()
        merged_data.update(self.data)
        merged_data.pop('refreshState', None)
        if not self.data.get('connectionInfo', None):
            raise OneViewModuleValueError('A connectionInfo field is required for this operation.')
        updated_san_manager = self.resource_client.update(resource=merged_data, id_or_uri=resource['uri'])
        return (True, self.MSG_UPDATED, updated_san_manager)