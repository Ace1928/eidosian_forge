from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v1 import notebooks_v1_messages as messages
class ProjectsLocationsExecutionsService(base_api.BaseApiService):
    """Service class for the projects_locations_executions resource."""
    _NAME = 'projects_locations_executions'

    def __init__(self, client):
        super(NotebooksV1.ProjectsLocationsExecutionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Execution in a given project and location.

      Args:
        request: (NotebooksProjectsLocationsExecutionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/executions', http_method='POST', method_id='notebooks.projects.locations.executions.create', ordered_params=['parent'], path_params=['parent'], query_params=['executionId'], relative_path='v1/{+parent}/executions', request_field='execution', request_type_name='NotebooksProjectsLocationsExecutionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes execution.

      Args:
        request: (NotebooksProjectsLocationsExecutionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/executions/{executionsId}', http_method='DELETE', method_id='notebooks.projects.locations.executions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NotebooksProjectsLocationsExecutionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of executions.

      Args:
        request: (NotebooksProjectsLocationsExecutionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Execution) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/executions/{executionsId}', http_method='GET', method_id='notebooks.projects.locations.executions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NotebooksProjectsLocationsExecutionsGetRequest', response_type_name='Execution', supports_download=False)

    def List(self, request, global_params=None):
        """Lists executions in a given project and location.

      Args:
        request: (NotebooksProjectsLocationsExecutionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListExecutionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/executions', http_method='GET', method_id='notebooks.projects.locations.executions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/executions', request_field='', request_type_name='NotebooksProjectsLocationsExecutionsListRequest', response_type_name='ListExecutionsResponse', supports_download=False)