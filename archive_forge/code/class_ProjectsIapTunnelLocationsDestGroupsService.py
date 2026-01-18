from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iap.v1 import iap_v1_messages as messages
class ProjectsIapTunnelLocationsDestGroupsService(base_api.BaseApiService):
    """Service class for the projects_iap_tunnel_locations_destGroups resource."""
    _NAME = 'projects_iap_tunnel_locations_destGroups'

    def __init__(self, client):
        super(IapV1.ProjectsIapTunnelLocationsDestGroupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new TunnelDestGroup.

      Args:
        request: (IapProjectsIapTunnelLocationsDestGroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TunnelDestGroup) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/iap_tunnel/locations/{locationsId}/destGroups', http_method='POST', method_id='iap.projects.iap_tunnel.locations.destGroups.create', ordered_params=['parent'], path_params=['parent'], query_params=['tunnelDestGroupId'], relative_path='v1/{+parent}/destGroups', request_field='tunnelDestGroup', request_type_name='IapProjectsIapTunnelLocationsDestGroupsCreateRequest', response_type_name='TunnelDestGroup', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a TunnelDestGroup.

      Args:
        request: (IapProjectsIapTunnelLocationsDestGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/iap_tunnel/locations/{locationsId}/destGroups/{destGroupsId}', http_method='DELETE', method_id='iap.projects.iap_tunnel.locations.destGroups.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IapProjectsIapTunnelLocationsDestGroupsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves an existing TunnelDestGroup.

      Args:
        request: (IapProjectsIapTunnelLocationsDestGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TunnelDestGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/iap_tunnel/locations/{locationsId}/destGroups/{destGroupsId}', http_method='GET', method_id='iap.projects.iap_tunnel.locations.destGroups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IapProjectsIapTunnelLocationsDestGroupsGetRequest', response_type_name='TunnelDestGroup', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the existing TunnelDestGroups. To group across all locations, use a `-` as the location ID. For example: `/v1/projects/123/iap_tunnel/locations/-/destGroups`.

      Args:
        request: (IapProjectsIapTunnelLocationsDestGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTunnelDestGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/iap_tunnel/locations/{locationsId}/destGroups', http_method='GET', method_id='iap.projects.iap_tunnel.locations.destGroups.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/destGroups', request_field='', request_type_name='IapProjectsIapTunnelLocationsDestGroupsListRequest', response_type_name='ListTunnelDestGroupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a TunnelDestGroup.

      Args:
        request: (IapProjectsIapTunnelLocationsDestGroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TunnelDestGroup) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/iap_tunnel/locations/{locationsId}/destGroups/{destGroupsId}', http_method='PATCH', method_id='iap.projects.iap_tunnel.locations.destGroups.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='tunnelDestGroup', request_type_name='IapProjectsIapTunnelLocationsDestGroupsPatchRequest', response_type_name='TunnelDestGroup', supports_download=False)