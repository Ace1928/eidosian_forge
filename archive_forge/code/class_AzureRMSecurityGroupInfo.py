from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMSecurityGroupInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(required=True, type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.name = None
        self.resource_group = None
        self.tags = None
        super(AzureRMSecurityGroupInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_securitygroup_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_securitygroup_facts' module has been renamed to 'azure_rm_securitygroup_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name is not None:
            info = self.get_item()
        else:
            info = self.list_items()
        if is_old_facts:
            self.results['ansible_facts'] = {'azure_securitygroups': info}
        self.results['securitygroups'] = info
        return self.results

    def get_item(self):
        self.log('Get properties for {0}'.format(self.name))
        item = None
        result = []
        try:
            item = self.network_client.network_security_groups.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        if item and self.has_tags(item.tags, self.tags):
            result = [create_network_security_group_dict(item)]
        return result

    def list_items(self):
        self.log('List all items')
        try:
            response = self.network_client.network_security_groups.list(self.resource_group)
        except Exception as exc:
            self.fail('Error listing all items - {0}'.format(str(exc)))
        results = []
        for item in response:
            if self.has_tags(item.tags, self.tags):
                results.append(create_network_security_group_dict(item))
        return results