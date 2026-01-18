from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
class OrganizationsLocationsFirewallEndpointsService(base_api.BaseApiService):
    """Service class for the organizations_locations_firewallEndpoints resource."""
    _NAME = 'organizations_locations_firewallEndpoints'

    def __init__(self, client):
        super(NetworksecurityV1.OrganizationsLocationsFirewallEndpointsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new FirewallEndpoint in a given project and location.

      Args:
        request: (NetworksecurityOrganizationsLocationsFirewallEndpointsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/firewallEndpoints', http_method='POST', method_id='networksecurity.organizations.locations.firewallEndpoints.create', ordered_params=['parent'], path_params=['parent'], query_params=['firewallEndpointId', 'requestId'], relative_path='v1/{+parent}/firewallEndpoints', request_field='firewallEndpoint', request_type_name='NetworksecurityOrganizationsLocationsFirewallEndpointsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Endpoint.

      Args:
        request: (NetworksecurityOrganizationsLocationsFirewallEndpointsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/firewallEndpoints/{firewallEndpointsId}', http_method='DELETE', method_id='networksecurity.organizations.locations.firewallEndpoints.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityOrganizationsLocationsFirewallEndpointsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Endpoint.

      Args:
        request: (NetworksecurityOrganizationsLocationsFirewallEndpointsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallEndpoint) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/firewallEndpoints/{firewallEndpointsId}', http_method='GET', method_id='networksecurity.organizations.locations.firewallEndpoints.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityOrganizationsLocationsFirewallEndpointsGetRequest', response_type_name='FirewallEndpoint', supports_download=False)

    def List(self, request, global_params=None):
        """Lists FirewallEndpoints in a given project and location.

      Args:
        request: (NetworksecurityOrganizationsLocationsFirewallEndpointsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFirewallEndpointsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/firewallEndpoints', http_method='GET', method_id='networksecurity.organizations.locations.firewallEndpoints.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/firewallEndpoints', request_field='', request_type_name='NetworksecurityOrganizationsLocationsFirewallEndpointsListRequest', response_type_name='ListFirewallEndpointsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a single Endpoint.

      Args:
        request: (NetworksecurityOrganizationsLocationsFirewallEndpointsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/firewallEndpoints/{firewallEndpointsId}', http_method='PATCH', method_id='networksecurity.organizations.locations.firewallEndpoints.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='firewallEndpoint', request_type_name='NetworksecurityOrganizationsLocationsFirewallEndpointsPatchRequest', response_type_name='Operation', supports_download=False)