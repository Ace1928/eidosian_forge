from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workstations.v1beta import workstations_v1beta_messages as messages
class ProjectsLocationsWorkstationClustersService(base_api.BaseApiService):
    """Service class for the projects_locations_workstationClusters resource."""
    _NAME = 'projects_locations_workstationClusters'

    def __init__(self, client):
        super(WorkstationsV1beta.ProjectsLocationsWorkstationClustersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new workstation cluster.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters', http_method='POST', method_id='workstations.projects.locations.workstationClusters.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly', 'workstationClusterId'], relative_path='v1beta/{+parent}/workstationClusters', request_field='workstationCluster', request_type_name='WorkstationsProjectsLocationsWorkstationClustersCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified workstation cluster.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}', http_method='DELETE', method_id='workstations.projects.locations.workstationClusters.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'force', 'validateOnly'], relative_path='v1beta/{+name}', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the requested workstation cluster.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkstationCluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}', http_method='GET', method_id='workstations.projects.locations.workstationClusters.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersGetRequest', response_type_name='WorkstationCluster', supports_download=False)

    def List(self, request, global_params=None):
        """Returns all workstation clusters in the specified location.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkstationClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters', http_method='GET', method_id='workstations.projects.locations.workstationClusters.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/workstationClusters', request_field='', request_type_name='WorkstationsProjectsLocationsWorkstationClustersListRequest', response_type_name='ListWorkstationClustersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing workstation cluster.

      Args:
        request: (WorkstationsProjectsLocationsWorkstationClustersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/workstationClusters/{workstationClustersId}', http_method='PATCH', method_id='workstations.projects.locations.workstationClusters.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask', 'validateOnly'], relative_path='v1beta/{+name}', request_field='workstationCluster', request_type_name='WorkstationsProjectsLocationsWorkstationClustersPatchRequest', response_type_name='Operation', supports_download=False)