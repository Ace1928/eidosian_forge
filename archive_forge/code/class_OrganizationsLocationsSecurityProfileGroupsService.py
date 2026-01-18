from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
class OrganizationsLocationsSecurityProfileGroupsService(base_api.BaseApiService):
    """Service class for the organizations_locations_securityProfileGroups resource."""
    _NAME = 'organizations_locations_securityProfileGroups'

    def __init__(self, client):
        super(NetworksecurityV1.OrganizationsLocationsSecurityProfileGroupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new SecurityProfileGroup in a given organization and location.

      Args:
        request: (NetworksecurityOrganizationsLocationsSecurityProfileGroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityProfileGroups', http_method='POST', method_id='networksecurity.organizations.locations.securityProfileGroups.create', ordered_params=['parent'], path_params=['parent'], query_params=['securityProfileGroupId'], relative_path='v1/{+parent}/securityProfileGroups', request_field='securityProfileGroup', request_type_name='NetworksecurityOrganizationsLocationsSecurityProfileGroupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single SecurityProfileGroup.

      Args:
        request: (NetworksecurityOrganizationsLocationsSecurityProfileGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityProfileGroups/{securityProfileGroupsId}', http_method='DELETE', method_id='networksecurity.organizations.locations.securityProfileGroups.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityOrganizationsLocationsSecurityProfileGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single SecurityProfileGroup.

      Args:
        request: (NetworksecurityOrganizationsLocationsSecurityProfileGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityProfileGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityProfileGroups/{securityProfileGroupsId}', http_method='GET', method_id='networksecurity.organizations.locations.securityProfileGroups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityOrganizationsLocationsSecurityProfileGroupsGetRequest', response_type_name='SecurityProfileGroup', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SecurityProfileGroups in a given organization and location.

      Args:
        request: (NetworksecurityOrganizationsLocationsSecurityProfileGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSecurityProfileGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityProfileGroups', http_method='GET', method_id='networksecurity.organizations.locations.securityProfileGroups.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/securityProfileGroups', request_field='', request_type_name='NetworksecurityOrganizationsLocationsSecurityProfileGroupsListRequest', response_type_name='ListSecurityProfileGroupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single SecurityProfileGroup.

      Args:
        request: (NetworksecurityOrganizationsLocationsSecurityProfileGroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/securityProfileGroups/{securityProfileGroupsId}', http_method='PATCH', method_id='networksecurity.organizations.locations.securityProfileGroups.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='securityProfileGroup', request_type_name='NetworksecurityOrganizationsLocationsSecurityProfileGroupsPatchRequest', response_type_name='Operation', supports_download=False)