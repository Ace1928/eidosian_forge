from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.file.v1beta1 import file_v1beta1_messages as messages
class ProjectsLocationsInstancesSharesService(base_api.BaseApiService):
    """Service class for the projects_locations_instances_shares resource."""
    _NAME = 'projects_locations_instances_shares'

    def __init__(self, client):
        super(FileV1beta1.ProjectsLocationsInstancesSharesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a share.

      Args:
        request: (FileProjectsLocationsInstancesSharesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/shares', http_method='POST', method_id='file.projects.locations.instances.shares.create', ordered_params=['parent'], path_params=['parent'], query_params=['shareId'], relative_path='v1beta1/{+parent}/shares', request_field='share', request_type_name='FileProjectsLocationsInstancesSharesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a share.

      Args:
        request: (FileProjectsLocationsInstancesSharesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/shares/{sharesId}', http_method='DELETE', method_id='file.projects.locations.instances.shares.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='FileProjectsLocationsInstancesSharesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a specific share.

      Args:
        request: (FileProjectsLocationsInstancesSharesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Share) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/shares/{sharesId}', http_method='GET', method_id='file.projects.locations.instances.shares.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='FileProjectsLocationsInstancesSharesGetRequest', response_type_name='Share', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all shares for a specified instance.

      Args:
        request: (FileProjectsLocationsInstancesSharesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSharesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/shares', http_method='GET', method_id='file.projects.locations.instances.shares.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/shares', request_field='', request_type_name='FileProjectsLocationsInstancesSharesListRequest', response_type_name='ListSharesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the settings of a specific share.

      Args:
        request: (FileProjectsLocationsInstancesSharesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/shares/{sharesId}', http_method='PATCH', method_id='file.projects.locations.instances.shares.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta1/{+name}', request_field='share', request_type_name='FileProjectsLocationsInstancesSharesPatchRequest', response_type_name='Operation', supports_download=False)