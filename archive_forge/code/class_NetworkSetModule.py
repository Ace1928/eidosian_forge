from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleResourceNotFound
class NetworkSetModule(OneViewModuleBase):
    MSG_CREATED = 'Network Set created successfully.'
    MSG_UPDATED = 'Network Set updated successfully.'
    MSG_DELETED = 'Network Set deleted successfully.'
    MSG_ALREADY_PRESENT = 'Network Set is already present.'
    MSG_ALREADY_ABSENT = 'Network Set is already absent.'
    MSG_ETHERNET_NETWORK_NOT_FOUND = 'Ethernet Network not found: '
    RESOURCE_FACT_NAME = 'network_set'
    argument_spec = dict(state=dict(default='present', choices=['present', 'absent']), data=dict(required=True, type='dict'))

    def __init__(self):
        super(NetworkSetModule, self).__init__(additional_arg_spec=self.argument_spec, validate_etag_support=True)
        self.resource_client = self.oneview_client.network_sets

    def execute_module(self):
        resource = self.get_by_name(self.data.get('name'))
        if self.state == 'present':
            return self._present(resource)
        elif self.state == 'absent':
            return self.resource_absent(resource)

    def _present(self, resource):
        scope_uris = self.data.pop('scopeUris', None)
        self._replace_network_name_by_uri(self.data)
        result = self.resource_present(resource, self.RESOURCE_FACT_NAME)
        if scope_uris is not None:
            result = self.resource_scopes_set(result, self.RESOURCE_FACT_NAME, scope_uris)
        return result

    def _get_ethernet_network_by_name(self, name):
        result = self.oneview_client.ethernet_networks.get_by('name', name)
        return result[0] if result else None

    def _get_network_uri(self, network_name_or_uri):
        if network_name_or_uri.startswith('/rest/ethernet-networks'):
            return network_name_or_uri
        else:
            enet_network = self._get_ethernet_network_by_name(network_name_or_uri)
            if enet_network:
                return enet_network['uri']
            else:
                raise OneViewModuleResourceNotFound(self.MSG_ETHERNET_NETWORK_NOT_FOUND + network_name_or_uri)

    def _replace_network_name_by_uri(self, data):
        if 'networkUris' in data:
            data['networkUris'] = [self._get_network_uri(x) for x in data['networkUris']]