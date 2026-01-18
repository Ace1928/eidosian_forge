from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.oslogin.v1beta import oslogin_v1beta_messages as messages
class UsersProjectsLocationsService(base_api.BaseApiService):
    """Service class for the users_projects_locations resource."""
    _NAME = 'users_projects_locations'

    def __init__(self, client):
        super(OsloginV1beta.UsersProjectsLocationsService, self).__init__(client)
        self._upload_configs = {}

    def SignSshPublicKey(self, request, global_params=None):
        """Signs an SSH public key for a user to authenticate to an instance.

      Args:
        request: (OsloginUsersProjectsLocationsSignSshPublicKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SignSshPublicKeyResponse) The response message.
      """
        config = self.GetMethodConfig('SignSshPublicKey')
        return self._RunMethod(config, request, global_params=global_params)
    SignSshPublicKey.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/users/{usersId}/projects/{projectsId}/locations/{locationsId}:signSshPublicKey', http_method='POST', method_id='oslogin.users.projects.locations.signSshPublicKey', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta/{+parent}:signSshPublicKey', request_field='signSshPublicKeyRequest', request_type_name='OsloginUsersProjectsLocationsSignSshPublicKeyRequest', response_type_name='SignSshPublicKeyResponse', supports_download=False)