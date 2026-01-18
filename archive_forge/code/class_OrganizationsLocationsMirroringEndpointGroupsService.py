from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1alpha1 import networksecurity_v1alpha1_messages as messages
class OrganizationsLocationsMirroringEndpointGroupsService(base_api.BaseApiService):
    """Service class for the organizations_locations_mirroringEndpointGroups resource."""
    _NAME = 'organizations_locations_mirroringEndpointGroups'

    def __init__(self, client):
        super(NetworksecurityV1alpha1.OrganizationsLocationsMirroringEndpointGroupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new MirroringEndpointGroup in a given organization and location.

      Args:
        request: (NetworksecurityOrganizationsLocationsMirroringEndpointGroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/locations/{locationsId}/mirroringEndpointGroups', http_method='POST', method_id='networksecurity.organizations.locations.mirroringEndpointGroups.create', ordered_params=['parent'], path_params=['parent'], query_params=['mirroringEndpointGroupId', 'requestId'], relative_path='v1alpha1/{+parent}/mirroringEndpointGroups', request_field='mirroringEndpointGroup', request_type_name='NetworksecurityOrganizationsLocationsMirroringEndpointGroupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single MirroringEndpointGroup.

      Args:
        request: (NetworksecurityOrganizationsLocationsMirroringEndpointGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/locations/{locationsId}/mirroringEndpointGroups/{mirroringEndpointGroupsId}', http_method='DELETE', method_id='networksecurity.organizations.locations.mirroringEndpointGroups.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityOrganizationsLocationsMirroringEndpointGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single MirroringEndpointGroup.

      Args:
        request: (NetworksecurityOrganizationsLocationsMirroringEndpointGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MirroringEndpointGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/locations/{locationsId}/mirroringEndpointGroups/{mirroringEndpointGroupsId}', http_method='GET', method_id='networksecurity.organizations.locations.mirroringEndpointGroups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworksecurityOrganizationsLocationsMirroringEndpointGroupsGetRequest', response_type_name='MirroringEndpointGroup', supports_download=False)

    def List(self, request, global_params=None):
        """Lists MirroringEndpointGroups in a given organization and location.

      Args:
        request: (NetworksecurityOrganizationsLocationsMirroringEndpointGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMirroringEndpointGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/locations/{locationsId}/mirroringEndpointGroups', http_method='GET', method_id='networksecurity.organizations.locations.mirroringEndpointGroups.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/mirroringEndpointGroups', request_field='', request_type_name='NetworksecurityOrganizationsLocationsMirroringEndpointGroupsListRequest', response_type_name='ListMirroringEndpointGroupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a single MirroringEndpointGroup.

      Args:
        request: (NetworksecurityOrganizationsLocationsMirroringEndpointGroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/locations/{locationsId}/mirroringEndpointGroups/{mirroringEndpointGroupsId}', http_method='PATCH', method_id='networksecurity.organizations.locations.mirroringEndpointGroups.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha1/{+name}', request_field='mirroringEndpointGroup', request_type_name='NetworksecurityOrganizationsLocationsMirroringEndpointGroupsPatchRequest', response_type_name='Operation', supports_download=False)