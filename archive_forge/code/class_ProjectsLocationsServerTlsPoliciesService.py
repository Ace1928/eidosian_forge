from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
class ProjectsLocationsServerTlsPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_serverTlsPolicies resource."""
    _NAME = 'projects_locations_serverTlsPolicies'

    def __init__(self, client):
        super(NetworksecurityV1.ProjectsLocationsServerTlsPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ServerTlsPolicy in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsServerTlsPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serverTlsPolicies', http_method='POST', method_id='networksecurity.projects.locations.serverTlsPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['serverTlsPolicyId'], relative_path='v1/{+parent}/serverTlsPolicies', request_field='serverTlsPolicy', request_type_name='NetworksecurityProjectsLocationsServerTlsPoliciesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ServerTlsPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsServerTlsPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serverTlsPolicies/{serverTlsPoliciesId}', http_method='DELETE', method_id='networksecurity.projects.locations.serverTlsPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsServerTlsPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ServerTlsPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsServerTlsPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServerTlsPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serverTlsPolicies/{serverTlsPoliciesId}', http_method='GET', method_id='networksecurity.projects.locations.serverTlsPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsServerTlsPoliciesGetRequest', response_type_name='ServerTlsPolicy', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworksecurityProjectsLocationsServerTlsPoliciesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serverTlsPolicies/{serverTlsPoliciesId}:getIamPolicy', http_method='GET', method_id='networksecurity.projects.locations.serverTlsPolicies.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='NetworksecurityProjectsLocationsServerTlsPoliciesGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ServerTlsPolicies in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsServerTlsPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServerTlsPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serverTlsPolicies', http_method='GET', method_id='networksecurity.projects.locations.serverTlsPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/serverTlsPolicies', request_field='', request_type_name='NetworksecurityProjectsLocationsServerTlsPoliciesListRequest', response_type_name='ListServerTlsPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single ServerTlsPolicy.

      Args:
        request: (NetworksecurityProjectsLocationsServerTlsPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serverTlsPolicies/{serverTlsPoliciesId}', http_method='PATCH', method_id='networksecurity.projects.locations.serverTlsPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='serverTlsPolicy', request_type_name='NetworksecurityProjectsLocationsServerTlsPoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworksecurityProjectsLocationsServerTlsPoliciesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serverTlsPolicies/{serverTlsPoliciesId}:setIamPolicy', http_method='POST', method_id='networksecurity.projects.locations.serverTlsPolicies.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='NetworksecurityProjectsLocationsServerTlsPoliciesSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworksecurityProjectsLocationsServerTlsPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serverTlsPolicies/{serverTlsPoliciesId}:testIamPermissions', http_method='POST', method_id='networksecurity.projects.locations.serverTlsPolicies.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='NetworksecurityProjectsLocationsServerTlsPoliciesTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)