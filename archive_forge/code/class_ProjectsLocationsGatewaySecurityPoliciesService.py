from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
class ProjectsLocationsGatewaySecurityPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_gatewaySecurityPolicies resource."""
    _NAME = 'projects_locations_gatewaySecurityPolicies'

    def __init__(self, client):
        super(NetworksecurityV1.ProjectsLocationsGatewaySecurityPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new GatewaySecurityPolicy in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsGatewaySecurityPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gatewaySecurityPolicies', http_method='POST', method_id='networksecurity.projects.locations.gatewaySecurityPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['gatewaySecurityPolicyId'], relative_path='v1/{+parent}/gatewaySecurityPolicies', request_field='gatewaySecurityPolicy', request_type_name='NetworksecurityProjectsLocationsGatewaySecurityPoliciesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single GatewaySecurityPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsGatewaySecurityPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gatewaySecurityPolicies/{gatewaySecurityPoliciesId}', http_method='DELETE', method_id='networksecurity.projects.locations.gatewaySecurityPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsGatewaySecurityPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single GatewaySecurityPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsGatewaySecurityPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GatewaySecurityPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gatewaySecurityPolicies/{gatewaySecurityPoliciesId}', http_method='GET', method_id='networksecurity.projects.locations.gatewaySecurityPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsGatewaySecurityPoliciesGetRequest', response_type_name='GatewaySecurityPolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists GatewaySecurityPolicies in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsGatewaySecurityPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGatewaySecurityPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gatewaySecurityPolicies', http_method='GET', method_id='networksecurity.projects.locations.gatewaySecurityPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/gatewaySecurityPolicies', request_field='', request_type_name='NetworksecurityProjectsLocationsGatewaySecurityPoliciesListRequest', response_type_name='ListGatewaySecurityPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single GatewaySecurityPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsGatewaySecurityPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gatewaySecurityPolicies/{gatewaySecurityPoliciesId}', http_method='PATCH', method_id='networksecurity.projects.locations.gatewaySecurityPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='gatewaySecurityPolicy', request_type_name='NetworksecurityProjectsLocationsGatewaySecurityPoliciesPatchRequest', response_type_name='Operation', supports_download=False)