from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudshell.v1alpha1 import cloudshell_v1alpha1_messages as messages
class UsersEnvironmentsPublicKeysService(base_api.BaseApiService):
    """Service class for the users_environments_publicKeys resource."""
    _NAME = 'users_environments_publicKeys'

    def __init__(self, client):
        super(CloudshellV1alpha1.UsersEnvironmentsPublicKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Adds a public SSH key to an environment, allowing clients with the corresponding private key to connect to that environment via SSH. If a key with the same format and content already exists, this will return the existing key.

      Args:
        request: (CloudshellUsersEnvironmentsPublicKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PublicKey) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/users/{usersId}/environments/{environmentsId}/publicKeys', http_method='POST', method_id='cloudshell.users.environments.publicKeys.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/publicKeys', request_field='createPublicKeyRequest', request_type_name='CloudshellUsersEnvironmentsPublicKeysCreateRequest', response_type_name='PublicKey', supports_download=False)

    def Delete(self, request, global_params=None):
        """Removes a public SSH key from an environment. Clients will no longer be able to connect to the environment using the corresponding private key.

      Args:
        request: (CloudshellUsersEnvironmentsPublicKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/users/{usersId}/environments/{environmentsId}/publicKeys/{publicKeysId}', http_method='DELETE', method_id='cloudshell.users.environments.publicKeys.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='CloudshellUsersEnvironmentsPublicKeysDeleteRequest', response_type_name='Empty', supports_download=False)