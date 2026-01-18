from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMAksAgentPoolVersion(AzureRMModuleBase):

    def __init__(self):
        self.module_args = dict(resource_group=dict(type='str', required=True), cluster_name=dict(type='str', required=True))
        self.results = dict(changed=False, azure_orchestrator_version=[])
        self.resource_group = None
        self.cluster_name = None
        super(AzureRMAksAgentPoolVersion, self).__init__(derived_arg_spec=self.module_args, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_args:
            setattr(self, key, kwargs[key])
        self.results['azure_orchestrator_version'] = self.get_all_versions()
        return self.results

    def get_all_versions(self):
        """
        Get all avaliable orchestrator version
        """
        try:
            result = list()
            response = self.managedcluster_client.agent_pools.get_available_agent_pool_versions(self.resource_group, self.cluster_name)
            orchestrators = response.agent_pool_versions
            for item in orchestrators:
                result.append(item.kubernetes_version)
            return result
        except Exception as exc:
            self.fail('Error when getting Agentpool supported orchestrator version list for locatio', exc)