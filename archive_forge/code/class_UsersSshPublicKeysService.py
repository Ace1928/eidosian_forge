from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.oslogin.v1beta import oslogin_v1beta_messages as messages
class UsersSshPublicKeysService(base_api.BaseApiService):
    """Service class for the users_sshPublicKeys resource."""
    _NAME = 'users_sshPublicKeys'

    def __init__(self, client):
        super(OsloginV1beta.UsersSshPublicKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an SSH public key.

      Args:
        request: (OsloginUsersSshPublicKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SshPublicKey) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/users/{usersId}/sshPublicKeys', http_method='POST', method_id='oslogin.users.sshPublicKeys.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta/{+parent}/sshPublicKeys', request_field='sshPublicKey', request_type_name='OsloginUsersSshPublicKeysCreateRequest', response_type_name='SshPublicKey', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an SSH public key.

      Args:
        request: (OsloginUsersSshPublicKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/users/{usersId}/sshPublicKeys/{sshPublicKeysId}', http_method='DELETE', method_id='oslogin.users.sshPublicKeys.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='OsloginUsersSshPublicKeysDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves an SSH public key.

      Args:
        request: (OsloginUsersSshPublicKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SshPublicKey) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/users/{usersId}/sshPublicKeys/{sshPublicKeysId}', http_method='GET', method_id='oslogin.users.sshPublicKeys.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='OsloginUsersSshPublicKeysGetRequest', response_type_name='SshPublicKey', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an SSH public key and returns the profile information. This method supports patch semantics.

      Args:
        request: (OsloginUsersSshPublicKeysPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SshPublicKey) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/users/{usersId}/sshPublicKeys/{sshPublicKeysId}', http_method='PATCH', method_id='oslogin.users.sshPublicKeys.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='sshPublicKey', request_type_name='OsloginUsersSshPublicKeysPatchRequest', response_type_name='SshPublicKey', supports_download=False)