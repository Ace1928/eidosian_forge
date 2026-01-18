from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
class ProjectsLocationsSnapshotSchedulePoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_snapshotSchedulePolicies resource."""
    _NAME = 'projects_locations_snapshotSchedulePolicies'

    def __init__(self, client):
        super(BaremetalsolutionV2.ProjectsLocationsSnapshotSchedulePoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a snapshot schedule policy in the specified project.

      Args:
        request: (BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SnapshotSchedulePolicy) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/snapshotSchedulePolicies', http_method='POST', method_id='baremetalsolution.projects.locations.snapshotSchedulePolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['snapshotSchedulePolicyId'], relative_path='v2/{+parent}/snapshotSchedulePolicies', request_field='snapshotSchedulePolicy', request_type_name='BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesCreateRequest', response_type_name='SnapshotSchedulePolicy', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a named snapshot schedule policy.

      Args:
        request: (BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/snapshotSchedulePolicies/{snapshotSchedulePoliciesId}', http_method='DELETE', method_id='baremetalsolution.projects.locations.snapshotSchedulePolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get details of a single snapshot schedule policy.

      Args:
        request: (BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SnapshotSchedulePolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/snapshotSchedulePolicies/{snapshotSchedulePoliciesId}', http_method='GET', method_id='baremetalsolution.projects.locations.snapshotSchedulePolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesGetRequest', response_type_name='SnapshotSchedulePolicy', supports_download=False)

    def List(self, request, global_params=None):
        """List snapshot schedule policies in a given project and location.

      Args:
        request: (BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSnapshotSchedulePoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/snapshotSchedulePolicies', http_method='GET', method_id='baremetalsolution.projects.locations.snapshotSchedulePolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/snapshotSchedulePolicies', request_field='', request_type_name='BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesListRequest', response_type_name='ListSnapshotSchedulePoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a snapshot schedule policy in the specified project.

      Args:
        request: (BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SnapshotSchedulePolicy) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/snapshotSchedulePolicies/{snapshotSchedulePoliciesId}', http_method='PATCH', method_id='baremetalsolution.projects.locations.snapshotSchedulePolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='snapshotSchedulePolicy', request_type_name='BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesPatchRequest', response_type_name='SnapshotSchedulePolicy', supports_download=False)