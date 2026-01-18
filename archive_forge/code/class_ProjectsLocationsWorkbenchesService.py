from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.seclm.v1alpha import seclm_v1alpha_messages as messages
class ProjectsLocationsWorkbenchesService(base_api.BaseApiService):
    """Service class for the projects_locations_workbenches resource."""
    _NAME = 'projects_locations_workbenches'

    def __init__(self, client):
        super(SeclmV1alpha.ProjectsLocationsWorkbenchesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new SecLM Workbench in a given project and location.

      Args:
        request: (SeclmProjectsLocationsWorkbenchesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/workbenches', http_method='POST', method_id='seclm.projects.locations.workbenches.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'workbenchId'], relative_path='v1alpha/{+parent}/workbenches', request_field='workbench', request_type_name='SeclmProjectsLocationsWorkbenchesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single SecLM Workbench.

      Args:
        request: (SeclmProjectsLocationsWorkbenchesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/workbenches/{workbenchesId}', http_method='DELETE', method_id='seclm.projects.locations.workbenches.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='SeclmProjectsLocationsWorkbenchesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single SecLM Workbench.

      Args:
        request: (SeclmProjectsLocationsWorkbenchesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Workbench) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/workbenches/{workbenchesId}', http_method='GET', method_id='seclm.projects.locations.workbenches.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='SeclmProjectsLocationsWorkbenchesGetRequest', response_type_name='Workbench', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SecLM Workbenches in a given project and location.

      Args:
        request: (SeclmProjectsLocationsWorkbenchesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkbenchesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/workbenches', http_method='GET', method_id='seclm.projects.locations.workbenches.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/workbenches', request_field='', request_type_name='SeclmProjectsLocationsWorkbenchesListRequest', response_type_name='ListWorkbenchesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single SecLM Workbench.

      Args:
        request: (SeclmProjectsLocationsWorkbenchesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/workbenches/{workbenchesId}', http_method='PATCH', method_id='seclm.projects.locations.workbenches.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='workbench', request_type_name='SeclmProjectsLocationsWorkbenchesPatchRequest', response_type_name='Operation', supports_download=False)

    def Query(self, request, global_params=None):
        """WorkbenchQuery is a custom pass-through verb that returns a single SecLM Workbench.

      Args:
        request: (SeclmProjectsLocationsWorkbenchesQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkbenchQueryResponse) The response message.
      """
        config = self.GetMethodConfig('Query')
        return self._RunMethod(config, request, global_params=global_params)
    Query.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/workbenches/{workbenchesId}:query', http_method='POST', method_id='seclm.projects.locations.workbenches.query', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:query', request_field='workbenchQueryRequest', request_type_name='SeclmProjectsLocationsWorkbenchesQueryRequest', response_type_name='WorkbenchQueryResponse', supports_download=False)