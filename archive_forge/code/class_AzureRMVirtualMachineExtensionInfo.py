from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMVirtualMachineExtensionInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), virtual_machine_name=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.virtual_machine_name = None
        self.name = None
        self.tags = None
        super(AzureRMVirtualMachineExtensionInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_virtualmachineextension_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_virtualmachineextension_facts' module has been renamed to 'azure_rm_virtualmachineextension_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name is not None:
            self.results['extensions'] = self.get_extensions()
        else:
            self.results['extensions'] = self.list_extensions()
        return self.results

    def get_extensions(self):
        response = None
        results = []
        try:
            response = self.compute_client.virtual_machine_extensions.get(resource_group_name=self.resource_group, vm_name=self.virtual_machine_name, vm_extension_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Virtual Machine Extension.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_response(response))
        return results

    def list_extensions(self):
        response = None
        results = []
        try:
            response = self.compute_client.virtual_machine_extensions.list(resource_group_name=self.resource_group, vm_name=self.virtual_machine_name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Virtual Machine Extension.')
        if response is not None and response.value is not None:
            for item in response.value:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_response(item))
        return results

    def format_response(self, item):
        d = item.as_dict()
        d = {'id': d.get('id', None), 'resource_group': self.resource_group, 'virtual_machine_name': self.virtual_machine_name, 'location': d.get('location'), 'name': d.get('name'), 'publisher': d.get('publisher'), 'type': d.get('type_properties_type'), 'settings': d.get('settings'), 'auto_upgrade_minor_version': d.get('auto_upgrade_minor_version'), 'tags': d.get('tags', None), 'provisioning_state': d.get('provisioning_state')}
        return d