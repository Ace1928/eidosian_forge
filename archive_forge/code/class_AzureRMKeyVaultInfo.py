from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMKeyVaultInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str'), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.resource_group = None
        self.name = None
        self.tags = None
        self.results = dict(changed=False)
        self._client = None
        super(AzureRMKeyVaultInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
        self._client = self.get_mgmt_svc_client(KeyVaultManagementClient, base_url=self._cloud_environment.endpoints.resource_manager, is_track2=True, api_version='2021-10-01')
        if self.name:
            if self.resource_group:
                self.results['keyvaults'] = self.get_by_name()
            else:
                self.fail('resource_group is required when filtering by name')
        elif self.resource_group:
            self.results['keyvaults'] = self.list_by_resource_group()
        else:
            self.results['keyvaults'] = self.list()
        return self.results

    def get_by_name(self):
        """
        Gets the properties of the specified key vault.

        :return: deserialized key vaultstate dictionary
        """
        self.log('Get the key vault {0}'.format(self.name))
        results = []
        try:
            response = self._client.vaults.get(resource_group_name=self.resource_group, vault_name=self.name)
            self.log('Response : {0}'.format(response))
            if response and self.has_tags(response.tags, self.tags):
                results.append(keyvault_to_dict(response))
        except ResourceNotFoundError as e:
            self.log('Did not find the key vault {0}: {1}'.format(self.name, str(e)))
        return results

    def list_by_resource_group(self):
        """
        Lists the properties of key vaults in specific resource group.

        :return: deserialized key vaults state dictionary
        """
        self.log('Get the key vaults in resource group {0}'.format(self.resource_group))
        results = []
        try:
            response = list(self._client.vaults.list_by_resource_group(resource_group_name=self.resource_group))
            self.log('Response : {0}'.format(response))
            if response:
                for item in response:
                    if self.has_tags(item.tags, self.tags):
                        results.append(keyvault_to_dict(item))
        except Exception as e:
            self.log('Did not find key vaults in resource group {0} : {1}.'.format(self.resource_group, str(e)))
        return results

    def list(self):
        """
        Lists the properties of key vaults in specific subscription.

        :return: deserialized key vaults state dictionary
        """
        self.log('Get the key vaults in current subscription')
        results = []
        try:
            response = list(self._client.vaults.list())
            self.log('Response : {0}'.format(response))
            if response:
                for item in response:
                    if self.has_tags(item.tags, self.tags):
                        source_id = item.id.split('/')
                        results.append(keyvault_to_dict(self._client.vaults.get(source_id[4], source_id[8])))
        except Exception as e:
            self.log('Did not find key vault in current subscription {0}.'.format(str(e)))
        return results