from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, azure_id_to_dict
from ansible.module_utils.common.dict_transformations import _camel_to_snake
class AzureRMRouteTableInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False, route_tables=[])
        self.name = None
        self.resource_group = None
        self.tags = None
        super(AzureRMRouteTableInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_routetable_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_routetable_facts' module has been renamed to 'azure_rm_routetable_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        response = []
        if self.name:
            response = self.get_item()
        elif self.resource_group:
            response = self.list_items()
        else:
            response = self.list_all_items()
        self.results['route_tables'] = [instance_to_dict(x) for x in response if self.has_tags(x.tags, self.tags)]
        return self.results

    def get_item(self):
        self.log('Get route table for {0}-{1}'.format(self.resource_group, self.name))
        try:
            item = self.network_client.route_tables.get(self.resource_group, self.name)
            return [item]
        except ResourceNotFoundError:
            pass
        return []

    def list_items(self):
        self.log('List all items in resource group')
        try:
            return self.network_client.route_tables.list(self.resource_group)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list items - {0}'.format(str(exc)))
        return []

    def list_all_items(self):
        self.log('List all items in subscription')
        try:
            return self.network_client.route_tables.list_all()
        except ResourceNotFoundError as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        return []