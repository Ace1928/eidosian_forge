from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
class ProjectsLocationsAuthorizationPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_authorizationPolicies resource."""
    _NAME = 'projects_locations_authorizationPolicies'

    def __init__(self, client):
        super(NetworksecurityV1.ProjectsLocationsAuthorizationPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new AuthorizationPolicy in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsAuthorizationPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/authorizationPolicies', http_method='POST', method_id='networksecurity.projects.locations.authorizationPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['authorizationPolicyId'], relative_path='v1/{+parent}/authorizationPolicies', request_field='authorizationPolicy', request_type_name='NetworksecurityProjectsLocationsAuthorizationPoliciesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single AuthorizationPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsAuthorizationPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/authorizationPolicies/{authorizationPoliciesId}', http_method='DELETE', method_id='networksecurity.projects.locations.authorizationPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsAuthorizationPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single AuthorizationPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsAuthorizationPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuthorizationPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/authorizationPolicies/{authorizationPoliciesId}', http_method='GET', method_id='networksecurity.projects.locations.authorizationPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsAuthorizationPoliciesGetRequest', response_type_name='AuthorizationPolicy', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworksecurityProjectsLocationsAuthorizationPoliciesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/authorizationPolicies/{authorizationPoliciesId}:getIamPolicy', http_method='GET', method_id='networksecurity.projects.locations.authorizationPolicies.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='NetworksecurityProjectsLocationsAuthorizationPoliciesGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists AuthorizationPolicies in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsAuthorizationPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAuthorizationPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/authorizationPolicies', http_method='GET', method_id='networksecurity.projects.locations.authorizationPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/authorizationPolicies', request_field='', request_type_name='NetworksecurityProjectsLocationsAuthorizationPoliciesListRequest', response_type_name='ListAuthorizationPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single AuthorizationPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsAuthorizationPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/authorizationPolicies/{authorizationPoliciesId}', http_method='PATCH', method_id='networksecurity.projects.locations.authorizationPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='authorizationPolicy', request_type_name='NetworksecurityProjectsLocationsAuthorizationPoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworksecurityProjectsLocationsAuthorizationPoliciesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/authorizationPolicies/{authorizationPoliciesId}:setIamPolicy', http_method='POST', method_id='networksecurity.projects.locations.authorizationPolicies.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='NetworksecurityProjectsLocationsAuthorizationPoliciesSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworksecurityProjectsLocationsAuthorizationPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/authorizationPolicies/{authorizationPoliciesId}:testIamPermissions', http_method='POST', method_id='networksecurity.projects.locations.authorizationPolicies.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='NetworksecurityProjectsLocationsAuthorizationPoliciesTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)