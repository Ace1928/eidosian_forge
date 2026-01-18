from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
class AzureRMDNSZoneInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False, ansible_info=dict(azure_dnszones=[]))
        self.name = None
        self.resource_group = None
        self.tags = None
        super(AzureRMDNSZoneInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_dnszone_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_dnszone_facts' module has been renamed to 'azure_rm_dnszone_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name and (not self.resource_group):
            self.fail('Parameter error: resource group required when filtering by name.')
        results = []
        if self.name is not None:
            results = self.get_item()
        elif self.resource_group:
            results = self.list_resource_group()
        else:
            results = self.list_items()
        self.results['ansible_info']['azure_dnszones'] = self.serialize_items(results)
        self.results['dnszones'] = self.curated_items(results)
        return self.results

    def get_item(self):
        self.log('Get properties for {0}'.format(self.name))
        item = None
        results = []
        try:
            item = self.dns_client.zones.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            results = [item]
        return results

    def list_resource_group(self):
        self.log('List items for resource group')
        try:
            response = self.dns_client.zones.list_by_resource_group(self.resource_group)
        except Exception as exc:
            self.fail('Failed to list for resource group {0} - {1}'.format(self.resource_group, str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item)
        return results

    def list_items(self):
        self.log('List all items')
        try:
            response = self.dns_client.zones.list()
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(item)
        return results

    def serialize_items(self, raws):
        return [self.serialize_obj(item, AZURE_OBJECT_CLASS) for item in raws] if raws else []

    def curated_items(self, raws):
        return [self.zone_to_dict(item) for item in raws] if raws else []

    def zone_to_dict(self, zone):
        return dict(id=zone.id, name=zone.name, number_of_record_sets=zone.number_of_record_sets, max_number_of_record_sets=zone.max_number_of_record_sets, name_servers=zone.name_servers, tags=zone.tags, type=zone.zone_type.lower(), registration_virtual_networks=[to_native(x.id) for x in zone.registration_virtual_networks] if zone.registration_virtual_networks else None, resolution_virtual_networks=[to_native(x.id) for x in zone.resolution_virtual_networks] if zone.resolution_virtual_networks else None)