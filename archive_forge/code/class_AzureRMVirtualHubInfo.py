from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
class AzureRMVirtualHubInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str'), name=dict(type='str'))
        self.resource_group = None
        self.name = None
        self.results = dict(changed=False)
        self.state = None
        self.status_code = [200]
        super(AzureRMVirtualHubInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.resource_group is not None and self.name is not None:
            self.results['virtual_hubs'] = self.format_item(self.get())
        elif self.resource_group is not None:
            self.results['virtual_hubs'] = self.format_item(self.list_by_resource_group())
        else:
            self.results['virtual_hubs'] = self.format_item(self.list())
        return self.results

    def get(self):
        response = None
        try:
            response = self.network_client.virtual_hubs.get(resource_group_name=self.resource_group, virtual_hub_name=self.name)
        except ResourceNotFoundError as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def list_by_resource_group(self):
        response = None
        try:
            response = self.network_client.virtual_hubs.list_by_resource_group(resource_group_name=self.resource_group)
        except ResourceNotFoundError as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def list(self):
        response = None
        try:
            response = self.network_client.virtual_hubs.list()
        except ResourceNotFoundError as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def format_item(self, item):
        if item is None:
            return None
        elif hasattr(item, 'as_dict'):
            return [item.as_dict()]
        else:
            result = []
            items = list(item)
            for tmp in items:
                result.append(tmp.as_dict())
            return result