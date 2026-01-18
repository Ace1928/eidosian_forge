from __future__ import absolute_import, division, print_function
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import (
class HashiVaultAuthMethodAzure(HashiVaultAuthMethodBase):
    """HashiVault auth method for Azure"""
    NAME = 'azure'
    OPTIONS = ['role_id', 'jwt', 'mount_point', 'azure_tenant_id', 'azure_client_id', 'azure_client_secret', 'azure_resource']

    def __init__(self, option_adapter, warning_callback, deprecate_callback):
        super(HashiVaultAuthMethodAzure, self).__init__(option_adapter, warning_callback, deprecate_callback)

    def validate(self):
        params = {'role': self._options.get_option_default('role_id'), 'jwt': self._options.get_option_default('jwt')}
        if not params['role']:
            raise HashiVaultValueError('role_id is required for azure authentication.')
        mount_point = self._options.get_option_default('mount_point')
        if mount_point:
            params['mount_point'] = mount_point
        if not params['jwt']:
            azure_tenant_id = self._options.get_option_default('azure_tenant_id')
            azure_client_id = self._options.get_option_default('azure_client_id')
            azure_client_secret = self._options.get_option_default('azure_client_secret')
            azure_resource = self._options.get_option('azure_resource')
            azure_scope = azure_resource + '/.default'
            try:
                import azure.identity
            except ImportError:
                raise HashiVaultValueError('azure-identity is required for getting access token from azure service principal or managed identity.')
            if azure_client_id and azure_client_secret:
                if not azure_tenant_id:
                    raise HashiVaultValueError('azure_tenant_id is required when using azure service principal.')
                azure_credentials = azure.identity.ClientSecretCredential(azure_tenant_id, azure_client_id, azure_client_secret)
            elif azure_client_id:
                azure_credentials = azure.identity.ManagedIdentityCredential(client_id=azure_client_id)
            else:
                azure_credentials = azure.identity.ManagedIdentityCredential()
            params['jwt'] = azure_credentials.get_token(azure_scope).token
        self._auth_azure_login_params = params

    def authenticate(self, client, use_token=True):
        params = self._auth_azure_login_params
        response = client.auth.azure.login(use_token=use_token, **params)
        return response