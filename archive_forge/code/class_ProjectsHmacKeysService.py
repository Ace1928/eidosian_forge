from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storage.v1 import storage_v1_messages as messages
class ProjectsHmacKeysService(base_api.BaseApiService):
    """Service class for the projects_hmacKeys resource."""
    _NAME = 'projects_hmacKeys'

    def __init__(self, client):
        super(StorageV1.ProjectsHmacKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new HMAC key for the specified service account.

      Args:
        request: (StorageProjectsHmacKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HmacKey) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='storage.projects.hmacKeys.create', ordered_params=['projectId', 'serviceAccountEmail'], path_params=['projectId'], query_params=['serviceAccountEmail', 'userProject'], relative_path='projects/{projectId}/hmacKeys', request_field='', request_type_name='StorageProjectsHmacKeysCreateRequest', response_type_name='HmacKey', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an HMAC key.

      Args:
        request: (StorageProjectsHmacKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StorageProjectsHmacKeysDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='storage.projects.hmacKeys.delete', ordered_params=['projectId', 'accessId'], path_params=['accessId', 'projectId'], query_params=['userProject'], relative_path='projects/{projectId}/hmacKeys/{accessId}', request_field='', request_type_name='StorageProjectsHmacKeysDeleteRequest', response_type_name='StorageProjectsHmacKeysDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves an HMAC key's metadata.

      Args:
        request: (StorageProjectsHmacKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HmacKeyMetadata) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='storage.projects.hmacKeys.get', ordered_params=['projectId', 'accessId'], path_params=['accessId', 'projectId'], query_params=['userProject'], relative_path='projects/{projectId}/hmacKeys/{accessId}', request_field='', request_type_name='StorageProjectsHmacKeysGetRequest', response_type_name='HmacKeyMetadata', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of HMAC keys matching the criteria.

      Args:
        request: (StorageProjectsHmacKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HmacKeysMetadata) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='storage.projects.hmacKeys.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['maxResults', 'pageToken', 'serviceAccountEmail', 'showDeletedKeys', 'userProject'], relative_path='projects/{projectId}/hmacKeys', request_field='', request_type_name='StorageProjectsHmacKeysListRequest', response_type_name='HmacKeysMetadata', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the state of an HMAC key. See the HMAC Key resource descriptor for valid states.

      Args:
        request: (StorageProjectsHmacKeysUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HmacKeyMetadata) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='storage.projects.hmacKeys.update', ordered_params=['projectId', 'accessId'], path_params=['accessId', 'projectId'], query_params=['userProject'], relative_path='projects/{projectId}/hmacKeys/{accessId}', request_field='hmacKeyMetadata', request_type_name='StorageProjectsHmacKeysUpdateRequest', response_type_name='HmacKeyMetadata', supports_download=False)