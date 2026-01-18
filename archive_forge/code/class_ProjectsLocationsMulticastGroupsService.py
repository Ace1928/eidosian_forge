from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsMulticastGroupsService(base_api.BaseApiService):
    """Service class for the projects_locations_multicastGroups resource."""
    _NAME = 'projects_locations_multicastGroups'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsMulticastGroupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new multicast group in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroups', http_method='POST', method_id='networkservices.projects.locations.multicastGroups.create', ordered_params=['parent'], path_params=['parent'], query_params=['multicastGroupId', 'requestId'], relative_path='v1/{+parent}/multicastGroups', request_field='multicastGroup', request_type_name='NetworkservicesProjectsLocationsMulticastGroupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single multicast group.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroups/{multicastGroupsId}', http_method='DELETE', method_id='networkservices.projects.locations.multicastGroups.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single multicast group.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MulticastGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroups/{multicastGroupsId}', http_method='GET', method_id='networkservices.projects.locations.multicastGroups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastGroupsGetRequest', response_type_name='MulticastGroup', supports_download=False)

    def List(self, request, global_params=None):
        """Lists multicast groups in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMulticastGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroups', http_method='GET', method_id='networkservices.projects.locations.multicastGroups.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/multicastGroups', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastGroupsListRequest', response_type_name='ListMulticastGroupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single multicast group.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastGroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastGroups/{multicastGroupsId}', http_method='PATCH', method_id='networkservices.projects.locations.multicastGroups.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='multicastGroup', request_type_name='NetworkservicesProjectsLocationsMulticastGroupsPatchRequest', response_type_name='Operation', supports_download=False)