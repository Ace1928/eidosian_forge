from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
class ProjectsLocationsTargetProjectsService(base_api.BaseApiService):
    """Service class for the projects_locations_targetProjects resource."""
    _NAME = 'projects_locations_targetProjects'

    def __init__(self, client):
        super(VmmigrationV1.ProjectsLocationsTargetProjectsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new TargetProject in a given project. NOTE: TargetProject is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (VmmigrationProjectsLocationsTargetProjectsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/targetProjects', http_method='POST', method_id='vmmigration.projects.locations.targetProjects.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'targetProjectId'], relative_path='v1/{+parent}/targetProjects', request_field='targetProject', request_type_name='VmmigrationProjectsLocationsTargetProjectsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single TargetProject. NOTE: TargetProject is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (VmmigrationProjectsLocationsTargetProjectsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/targetProjects/{targetProjectsId}', http_method='DELETE', method_id='vmmigration.projects.locations.targetProjects.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsTargetProjectsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single TargetProject. NOTE: TargetProject is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (VmmigrationProjectsLocationsTargetProjectsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetProject) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/targetProjects/{targetProjectsId}', http_method='GET', method_id='vmmigration.projects.locations.targetProjects.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsTargetProjectsGetRequest', response_type_name='TargetProject', supports_download=False)

    def List(self, request, global_params=None):
        """Lists TargetProjects in a given project. NOTE: TargetProject is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (VmmigrationProjectsLocationsTargetProjectsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTargetProjectsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/targetProjects', http_method='GET', method_id='vmmigration.projects.locations.targetProjects.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/targetProjects', request_field='', request_type_name='VmmigrationProjectsLocationsTargetProjectsListRequest', response_type_name='ListTargetProjectsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single TargetProject. NOTE: TargetProject is a global resource; hence the only supported value for location is `global`.

      Args:
        request: (VmmigrationProjectsLocationsTargetProjectsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/targetProjects/{targetProjectsId}', http_method='PATCH', method_id='vmmigration.projects.locations.targetProjects.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='targetProject', request_type_name='VmmigrationProjectsLocationsTargetProjectsPatchRequest', response_type_name='Operation', supports_download=False)