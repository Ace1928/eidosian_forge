from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsWasmPluginsService(base_api.BaseApiService):
    """Service class for the projects_locations_wasmPlugins resource."""
    _NAME = 'projects_locations_wasmPlugins'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsWasmPluginsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `WasmPlugin` resource in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsWasmPluginsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/wasmPlugins', http_method='POST', method_id='networkservices.projects.locations.wasmPlugins.create', ordered_params=['parent'], path_params=['parent'], query_params=['wasmPluginId'], relative_path='v1/{+parent}/wasmPlugins', request_field='wasmPlugin', request_type_name='NetworkservicesProjectsLocationsWasmPluginsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified `WasmPlugin` resource.

      Args:
        request: (NetworkservicesProjectsLocationsWasmPluginsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/wasmPlugins/{wasmPluginsId}', http_method='DELETE', method_id='networkservices.projects.locations.wasmPlugins.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsWasmPluginsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of the specified `WasmPlugin` resource.

      Args:
        request: (NetworkservicesProjectsLocationsWasmPluginsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WasmPlugin) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/wasmPlugins/{wasmPluginsId}', http_method='GET', method_id='networkservices.projects.locations.wasmPlugins.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsWasmPluginsGetRequest', response_type_name='WasmPlugin', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `WasmPlugin` resources in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsWasmPluginsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWasmPluginsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/wasmPlugins', http_method='GET', method_id='networkservices.projects.locations.wasmPlugins.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/wasmPlugins', request_field='', request_type_name='NetworkservicesProjectsLocationsWasmPluginsListRequest', response_type_name='ListWasmPluginsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of the specified `WasmPlugin` resource.

      Args:
        request: (NetworkservicesProjectsLocationsWasmPluginsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/wasmPlugins/{wasmPluginsId}', http_method='PATCH', method_id='networkservices.projects.locations.wasmPlugins.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='wasmPlugin', request_type_name='NetworkservicesProjectsLocationsWasmPluginsPatchRequest', response_type_name='Operation', supports_download=False)