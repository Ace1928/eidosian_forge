from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apikeys.v2 import apikeys_v2_messages as messages
class ProjectsLocationsKeysService(base_api.BaseApiService):
    """Service class for the projects_locations_keys resource."""
    _NAME = 'projects_locations_keys'

    def __init__(self, client):
        super(ApikeysV2.ProjectsLocationsKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new API key. NOTE: Key is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (ApikeysProjectsLocationsKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/keys', http_method='POST', method_id='apikeys.projects.locations.keys.create', ordered_params=['parent'], path_params=['parent'], query_params=['keyId'], relative_path='v2/{+parent}/keys', request_field='v2Key', request_type_name='ApikeysProjectsLocationsKeysCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an API key. Deleted key can be retrieved within 30 days of deletion. Afterward, key will be purged from the project. NOTE: Key is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (ApikeysProjectsLocationsKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/keys/{keysId}', http_method='DELETE', method_id='apikeys.projects.locations.keys.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v2/{+name}', request_field='', request_type_name='ApikeysProjectsLocationsKeysDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the metadata for an API key. The key string of the API key isn't included in the response. NOTE: Key is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (ApikeysProjectsLocationsKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (V2Key) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/keys/{keysId}', http_method='GET', method_id='apikeys.projects.locations.keys.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='ApikeysProjectsLocationsKeysGetRequest', response_type_name='V2Key', supports_download=False)

    def GetKeyString(self, request, global_params=None):
        """Get the key string for an API key. NOTE: Key is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (ApikeysProjectsLocationsKeysGetKeyStringRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (V2GetKeyStringResponse) The response message.
      """
        config = self.GetMethodConfig('GetKeyString')
        return self._RunMethod(config, request, global_params=global_params)
    GetKeyString.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/keys/{keysId}/keyString', http_method='GET', method_id='apikeys.projects.locations.keys.getKeyString', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}/keyString', request_field='', request_type_name='ApikeysProjectsLocationsKeysGetKeyStringRequest', response_type_name='V2GetKeyStringResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the API keys owned by a project. The key string of the API key isn't included in the response. NOTE: Key is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (ApikeysProjectsLocationsKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (V2ListKeysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/keys', http_method='GET', method_id='apikeys.projects.locations.keys.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v2/{+parent}/keys', request_field='', request_type_name='ApikeysProjectsLocationsKeysListRequest', response_type_name='V2ListKeysResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the modifiable fields of an API key. The key string of the API key isn't included in the response. NOTE: Key is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (ApikeysProjectsLocationsKeysPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/keys/{keysId}', http_method='PATCH', method_id='apikeys.projects.locations.keys.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='v2Key', request_type_name='ApikeysProjectsLocationsKeysPatchRequest', response_type_name='Operation', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes an API key which was deleted within 30 days. NOTE: Key is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (ApikeysProjectsLocationsKeysUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/keys/{keysId}:undelete', http_method='POST', method_id='apikeys.projects.locations.keys.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:undelete', request_field='v2UndeleteKeyRequest', request_type_name='ApikeysProjectsLocationsKeysUndeleteRequest', response_type_name='Operation', supports_download=False)