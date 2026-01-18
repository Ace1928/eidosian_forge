from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.mediaasset.v1alpha import mediaasset_v1alpha_messages as messages
class ProjectsLocationsModulesService(base_api.BaseApiService):
    """Service class for the projects_locations_modules resource."""
    _NAME = 'projects_locations_modules'

    def __init__(self, client):
        super(MediaassetV1alpha.ProjectsLocationsModulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new module in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsModulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/modules', http_method='POST', method_id='mediaasset.projects.locations.modules.create', ordered_params=['parent'], path_params=['parent'], query_params=['moduleId'], relative_path='v1alpha/{+parent}/modules', request_field='module', request_type_name='MediaassetProjectsLocationsModulesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single module.

      Args:
        request: (MediaassetProjectsLocationsModulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/modules/{modulesId}', http_method='DELETE', method_id='mediaasset.projects.locations.modules.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsModulesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single module.

      Args:
        request: (MediaassetProjectsLocationsModulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Module) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/modules/{modulesId}', http_method='GET', method_id='mediaasset.projects.locations.modules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsModulesGetRequest', response_type_name='Module', supports_download=False)

    def List(self, request, global_params=None):
        """Lists modules in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsModulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListModulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/modules', http_method='GET', method_id='mediaasset.projects.locations.modules.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/modules', request_field='', request_type_name='MediaassetProjectsLocationsModulesListRequest', response_type_name='ListModulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single module.

      Args:
        request: (MediaassetProjectsLocationsModulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/modules/{modulesId}', http_method='PATCH', method_id='mediaasset.projects.locations.modules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha/{+name}', request_field='module', request_type_name='MediaassetProjectsLocationsModulesPatchRequest', response_type_name='Operation', supports_download=False)