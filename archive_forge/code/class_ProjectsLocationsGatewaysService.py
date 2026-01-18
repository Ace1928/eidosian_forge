from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigateway.v1beta import apigateway_v1beta_messages as messages
class ProjectsLocationsGatewaysService(base_api.BaseApiService):
    """Service class for the projects_locations_gateways resource."""
    _NAME = 'projects_locations_gateways'

    def __init__(self, client):
        super(ApigatewayV1beta.ProjectsLocationsGatewaysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Gateway in a given project and location.

      Args:
        request: (ApigatewayProjectsLocationsGatewaysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApigatewayOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/gateways', http_method='POST', method_id='apigateway.projects.locations.gateways.create', ordered_params=['parent'], path_params=['parent'], query_params=['gatewayId'], relative_path='v1beta/{+parent}/gateways', request_field='apigatewayGateway', request_type_name='ApigatewayProjectsLocationsGatewaysCreateRequest', response_type_name='ApigatewayOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Gateway.

      Args:
        request: (ApigatewayProjectsLocationsGatewaysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApigatewayOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/gateways/{gatewaysId}', http_method='DELETE', method_id='apigateway.projects.locations.gateways.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='ApigatewayProjectsLocationsGatewaysDeleteRequest', response_type_name='ApigatewayOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Gateway.

      Args:
        request: (ApigatewayProjectsLocationsGatewaysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApigatewayGateway) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/gateways/{gatewaysId}', http_method='GET', method_id='apigateway.projects.locations.gateways.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='ApigatewayProjectsLocationsGatewaysGetRequest', response_type_name='ApigatewayGateway', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (ApigatewayProjectsLocationsGatewaysGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApigatewayPolicy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/gateways/{gatewaysId}:getIamPolicy', http_method='GET', method_id='apigateway.projects.locations.gateways.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1beta/{+resource}:getIamPolicy', request_field='', request_type_name='ApigatewayProjectsLocationsGatewaysGetIamPolicyRequest', response_type_name='ApigatewayPolicy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Gateways in a given project and location.

      Args:
        request: (ApigatewayProjectsLocationsGatewaysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApigatewayListGatewaysResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/gateways', http_method='GET', method_id='apigateway.projects.locations.gateways.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/gateways', request_field='', request_type_name='ApigatewayProjectsLocationsGatewaysListRequest', response_type_name='ApigatewayListGatewaysResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Gateway.

      Args:
        request: (ApigatewayProjectsLocationsGatewaysPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApigatewayOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/gateways/{gatewaysId}', http_method='PATCH', method_id='apigateway.projects.locations.gateways.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='apigatewayGateway', request_type_name='ApigatewayProjectsLocationsGatewaysPatchRequest', response_type_name='ApigatewayOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (ApigatewayProjectsLocationsGatewaysSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApigatewayPolicy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/gateways/{gatewaysId}:setIamPolicy', http_method='POST', method_id='apigateway.projects.locations.gateways.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:setIamPolicy', request_field='apigatewaySetIamPolicyRequest', request_type_name='ApigatewayProjectsLocationsGatewaysSetIamPolicyRequest', response_type_name='ApigatewayPolicy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (ApigatewayProjectsLocationsGatewaysTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApigatewayTestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/gateways/{gatewaysId}:testIamPermissions', http_method='POST', method_id='apigateway.projects.locations.gateways.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:testIamPermissions', request_field='apigatewayTestIamPermissionsRequest', request_type_name='ApigatewayProjectsLocationsGatewaysTestIamPermissionsRequest', response_type_name='ApigatewayTestIamPermissionsResponse', supports_download=False)