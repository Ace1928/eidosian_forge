from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.batch.v1alpha import batch_v1alpha_messages as messages
class ProjectsLocationsResourceAllowancesService(base_api.BaseApiService):
    """Service class for the projects_locations_resourceAllowances resource."""
    _NAME = 'projects_locations_resourceAllowances'

    def __init__(self, client):
        super(BatchV1alpha.ProjectsLocationsResourceAllowancesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a Resource Allowance.

      Args:
        request: (BatchProjectsLocationsResourceAllowancesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourceAllowance) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceAllowances', http_method='POST', method_id='batch.projects.locations.resourceAllowances.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'resourceAllowanceId'], relative_path='v1alpha/{+parent}/resourceAllowances', request_field='resourceAllowance', request_type_name='BatchProjectsLocationsResourceAllowancesCreateRequest', response_type_name='ResourceAllowance', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a ResourceAllowance.

      Args:
        request: (BatchProjectsLocationsResourceAllowancesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceAllowances/{resourceAllowancesId}', http_method='DELETE', method_id='batch.projects.locations.resourceAllowances.delete', ordered_params=['name'], path_params=['name'], query_params=['reason', 'requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='BatchProjectsLocationsResourceAllowancesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Get a ResourceAllowance specified by its resource name.

      Args:
        request: (BatchProjectsLocationsResourceAllowancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourceAllowance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceAllowances/{resourceAllowancesId}', http_method='GET', method_id='batch.projects.locations.resourceAllowances.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='BatchProjectsLocationsResourceAllowancesGetRequest', response_type_name='ResourceAllowance', supports_download=False)

    def List(self, request, global_params=None):
        """List all ResourceAllowances for a project within a region.

      Args:
        request: (BatchProjectsLocationsResourceAllowancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListResourceAllowancesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceAllowances', http_method='GET', method_id='batch.projects.locations.resourceAllowances.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/resourceAllowances', request_field='', request_type_name='BatchProjectsLocationsResourceAllowancesListRequest', response_type_name='ListResourceAllowancesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a Resource Allowance.

      Args:
        request: (BatchProjectsLocationsResourceAllowancesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourceAllowance) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceAllowances/{resourceAllowancesId}', http_method='PATCH', method_id='batch.projects.locations.resourceAllowances.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='resourceAllowance', request_type_name='BatchProjectsLocationsResourceAllowancesPatchRequest', response_type_name='ResourceAllowance', supports_download=False)