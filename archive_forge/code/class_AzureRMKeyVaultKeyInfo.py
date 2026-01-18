from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMKeyVaultKeyInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(version=dict(type='str', default='current'), name=dict(type='str'), vault_uri=dict(type='str', required=True), show_deleted_key=dict(type='bool', default=False), tags=dict(type='list', elements='str'))
        self.vault_uri = None
        self.name = None
        self.version = None
        self.show_deleted_key = False
        self.tags = None
        self.results = dict(changed=False)
        self._client = None
        super(AzureRMKeyVaultKeyInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
        self._client = self.get_keyvault_client()
        if self.name:
            if self.show_deleted_key:
                self.results['keys'] = self.get_deleted_key()
            elif self.version == 'all':
                self.results['keys'] = self.get_key_versions()
            else:
                self.results['keys'] = self.get_key()
        elif self.show_deleted_key:
            self.results['keys'] = self.list_deleted_keys()
        else:
            self.results['keys'] = self.list_keys()
        return self.results

    def get_keyvault_client(self):
        return KeyClient(vault_url=self.vault_uri, credential=self.azure_auth.azure_credential_track2)

    def get_key(self):
        """
        Gets the properties of the specified key in key vault.

        :return: deserialized key state dictionary
        """
        self.log('Get the key {0}'.format(self.name))
        results = []
        try:
            if self.version == 'current':
                response = self._client.get_key(name=self.name, version=None)
            else:
                response = self._client.get_key(name=self.name, version=self.version)
            if response:
                response = keybundle_to_dict(response)
                if self.has_tags(response['tags'], self.tags):
                    self.log('Response : {0}'.format(response))
                    results.append(response)
        except Exception as e:
            self.fail(e)
            self.log('Did not find the key vault key {0}: {1}'.format(self.name, str(e)))
        return results

    def get_key_versions(self):
        """
        Lists keys versions.

        :return: deserialized versions of key, includes key identifier, attributes and tags
        """
        self.log('Get the key versions {0}'.format(self.name))
        results = []
        try:
            response = self._client.list_properties_of_key_versions(name=self.name)
            self.log('Response : {0}'.format(response))
            if response:
                for item in response:
                    item = keyitem_to_dict(item)
                    if self.has_tags(item['tags'], self.tags):
                        results.append(item)
        except Exception as e:
            self.log('Did not find key versions {0} : {1}.'.format(self.name, str(e)))
        return results

    def list_keys(self):
        """
        Lists keys in specific key vault.

        :return: deserialized keys, includes key identifier, attributes and tags.
        """
        self.log('Get the key vaults in current subscription')
        results = []
        try:
            response = self._client.list_properties_of_keys()
            self.log('Response : {0}'.format(response))
            if response:
                for item in response:
                    item = keyitem_to_dict(item)
                    if self.has_tags(item['tags'], self.tags):
                        results.append(item)
        except Exception as e:
            self.log('Did not find key vault in current subscription {0}.'.format(str(e)))
        return results

    def get_deleted_key(self):
        """
        Gets the properties of the specified deleted key in key vault.

        :return: deserialized key state dictionary
        """
        self.log('Get the key {0}'.format(self.name))
        results = []
        try:
            response = self._client.get_deleted_key(name=self.name)
            if response:
                response = deletedkeybundle_to_dict(response)
                if self.has_tags(response['tags'], self.tags):
                    self.log('Response : {0}'.format(response))
                    results.append(response)
        except Exception as e:
            self.log('Did not find the key vault key {0}: {1}'.format(self.name, str(e)))
        return results

    def list_deleted_keys(self):
        """
        Lists deleted keys in specific key vault.

        :return: deserialized keys, includes key identifier, attributes and tags.
        """
        self.log('Get the key vaults in current subscription')
        results = []
        try:
            response = self._client.list_deleted_keys()
            self.log('Response : {0}'.format(response))
            if response:
                for item in response:
                    item = deletedkeyitem_to_dict(item)
                    if self.has_tags(item['tags'], self.tags):
                        results.append(item)
        except Exception as e:
            self.log('Did not find key vault in current subscription {0}.'.format(str(e)))
        return results