from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMAKSVersion(AzureRMModuleBase):

    def __init__(self):
        self.module_args = dict(location=dict(type='str', required=True), version=dict(type='str'))
        self.results = dict(changed=False, azure_aks_versions=[])
        self.location = None
        self.version = None
        super(AzureRMAKSVersion, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_aksversion_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_aksversion_facts' module has been renamed to 'azure_rm_aksversion_info'", version=(2.9,))
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        self.results['azure_aks_versions'] = self.get_all_versions(self.location, self.version)
        return self.results

    def get_all_versions(self, location, version):
        """
        Get all kubernetes version supported by AKS
        :return: ordered version list
        """
        try:
            result = dict()
            response = self.containerservice_client.container_services.list_orchestrators(self.location, resource_type='managedClusters')
            orchestrators = response.orchestrators
            for item in orchestrators:
                result[item.orchestrator_version] = [x.orchestrator_version for x in item.upgrades] if item.upgrades else []
            if version:
                return result.get(version) or []
            else:
                keys = list(result.keys())
                keys.sort()
                return keys
        except Exception as exc:
            self.fail('Error when getting AKS supported kubernetes version list for location {0} - {1}'.format(self.location, exc.message or str(exc)))