from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMResourceGroupInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), tags=dict(type='list', elements='str'), list_resources=dict(type='bool'))
        self.results = dict(changed=False, resourcegroups=[])
        self.name = None
        self.tags = None
        self.list_resources = None
        super(AzureRMResourceGroupInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_resourcegroup_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_resourcegroup_facts' module has been renamed to 'azure_rm_resourcegroup_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name:
            result = self.get_item()
        else:
            result = self.list_items()
        if self.list_resources:
            for item in result:
                item['resources'] = self.list_by_rg(item['name'])
        if is_old_facts:
            self.results['ansible_facts']['azure_resourcegroups'] = result
        self.results['resourcegroups'] = result
        return self.results

    def get_item(self):
        self.log('Get properties for {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self.rm_client.resource_groups.get(self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            result = [self.serialize_obj(item, AZURE_OBJECT_CLASS)]
        return result

    def list_items(self):
        self.log('List all items')
        try:
            response = self.rm_client.resource_groups.list()
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(self.serialize_obj(item, AZURE_OBJECT_CLASS))
        return results

    def list_by_rg(self, name):
        self.log('List resources under resource group')
        results = []
        try:
            response = self.rm_client.resources.list_by_resource_group(name)
            while True:
                results.append(response.next().as_dict())
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Error when listing resources under resource group {0}: {1}'.format(name, exc.message or str(exc)))
        return results