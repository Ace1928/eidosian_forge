from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMAvailabilitySetInfo(AzureRMModuleBase):
    """Utility class to get availability set facts"""

    def __init__(self):
        self.module_args = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False, ansible_info=dict(azure_availabilitysets=[]))
        self.name = None
        self.resource_group = None
        self.tags = None
        super(AzureRMAvailabilitySetInfo, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_availabilityset_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_availabilityset_facts' module has been renamed to 'azure_rm_availabilityset_info'", version=(2.9,))
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        if self.name and (not self.resource_group):
            self.fail('Parameter error: resource group required when filtering by name.')
        if self.name:
            self.results['ansible_info']['azure_availabilitysets'] = self.get_item()
        else:
            self.results['ansible_info']['azure_availabilitysets'] = self.list_items()
        return self.results

    def get_item(self):
        """Get a single availability set"""
        self.log('Get properties for {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self.compute_client.availability_sets.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            avase = self.serialize_obj(item, AZURE_OBJECT_CLASS)
            avase['name'] = item.name
            avase['type'] = item.type
            avase['sku'] = item.sku.name
            result = [avase]
        return result

    def list_items(self):
        """Get all availability sets"""
        self.log('List all availability sets')
        try:
            response = self.compute_client.availability_sets.list(self.resource_group)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                avase = self.serialize_obj(item, AZURE_OBJECT_CLASS)
                avase['name'] = item.name
                avase['type'] = item.type
                avase['sku'] = item.sku.name
                results.append(avase)
        return results