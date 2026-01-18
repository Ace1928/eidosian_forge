from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
class AzureRMAutoScaleInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict()
        self.resource_group = None
        self.name = None
        self.tags = None
        super(AzureRMAutoScaleInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_autoscale_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_autoscale_facts' module has been renamed to 'azure_rm_autoscale_info'", version=(2.9,))
        for key in list(self.module_arg_spec):
            setattr(self, key, kwargs[key])
        if self.resource_group and self.name:
            self.results['autoscales'] = self.get()
        elif self.resource_group:
            self.results['autoscales'] = self.list_by_resource_group()
        return self.results

    def get(self):
        result = []
        try:
            instance = self.monitor_autoscale_settings_client.autoscale_settings.get(self.resource_group, self.name)
            result = [auto_scale_to_dict(instance) if self.has_tags(instance.tags, self.tags) else None]
        except Exception as ex:
            self.log('Could not get facts for autoscale {0} - {1}.'.format(self.name, str(ex)))
        return result

    def list_by_resource_group(self):
        results = []
        try:
            response = self.monitor_autoscale_settings_client.autoscale_settings.list_by_resource_group(self.resource_group)
            results = [auto_scale_to_dict(item) for item in response if self.has_tags(item.tags, self.tags)]
        except Exception as ex:
            self.log('Could not get facts for autoscale {0} - {1}.'.format(self.name, str(ex)))
        return results