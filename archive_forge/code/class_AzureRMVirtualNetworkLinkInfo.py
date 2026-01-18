from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMVirtualNetworkLinkInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str', required=True), zone_name=dict(type='str', required=True), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.name = None
        self.resource_group = None
        self.zone_name = None
        self.tags = None
        self.log_path = None
        self.log_mode = None
        super(AzureRMVirtualNetworkLinkInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        results = []
        if self.name is not None:
            results = self.get_item()
        else:
            results = self.list_items()
        self.results['virtualnetworklinks'] = self.curated_items(results)
        return self.results

    def get_item(self):
        self.log('Get properties for {0}'.format(self.name))
        item = None
        results = []
        try:
            item = self.private_dns_client.virtual_network_links.get(self.resource_group, self.zone_name, self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            results = [item]
        return results

    def list_items(self):
        self.log('List all virtual network links for private DNS zone - {0}'.format(self.zone_name))
        try:
            response = self.private_dns_client.virtual_network_links.list(self.resource_group, self.zone_name)
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item)
        return results

    def curated_items(self, raws):
        return [self.vnetlink_to_dict(item) for item in raws] if raws else []

    def vnetlink_to_dict(self, link):
        result = dict(id=link.id, name=link.name, virtual_network=dict(id=link.virtual_network.id), registration_enabled=link.registration_enabled, tags=link.tags, virtual_network_link_state=link.virtual_network_link_state, provisioning_state=link.provisioning_state)
        return result