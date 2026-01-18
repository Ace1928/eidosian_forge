from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleValueError
def _get_provider_uri_by_display_name(self, data):
    display_name = data.get('providerDisplayName')
    provider_uri = self.resource_client.get_provider_uri(display_name)
    if not provider_uri:
        raise OneViewModuleValueError(self.MSG_SAN_MANAGER_PROVIDER_DISPLAY_NAME_NOT_FOUND.format(display_name))
    return provider_uri