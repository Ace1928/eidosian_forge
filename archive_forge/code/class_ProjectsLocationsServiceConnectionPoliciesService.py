from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1 import networkconnectivity_v1_messages as messages
class ProjectsLocationsServiceConnectionPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_serviceConnectionPolicies resource."""
    _NAME = 'projects_locations_serviceConnectionPolicies'

    def __init__(self, client):
        super(NetworkconnectivityV1.ProjectsLocationsServiceConnectionPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ServiceConnectionPolicy in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionPolicies', http_method='POST', method_id='networkconnectivity.projects.locations.serviceConnectionPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'serviceConnectionPolicyId'], relative_path='v1/{+parent}/serviceConnectionPolicies', request_field='serviceConnectionPolicy', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionPoliciesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ServiceConnectionPolicy.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionPolicies/{serviceConnectionPoliciesId}', http_method='DELETE', method_id='networkconnectivity.projects.locations.serviceConnectionPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionPoliciesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ServiceConnectionPolicy.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceConnectionPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionPolicies/{serviceConnectionPoliciesId}', http_method='GET', method_id='networkconnectivity.projects.locations.serviceConnectionPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionPoliciesGetRequest', response_type_name='ServiceConnectionPolicy', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionPoliciesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionPolicies/{serviceConnectionPoliciesId}:getIamPolicy', http_method='GET', method_id='networkconnectivity.projects.locations.serviceConnectionPolicies.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionPoliciesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ServiceConnectionPolicies in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceConnectionPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionPolicies', http_method='GET', method_id='networkconnectivity.projects.locations.serviceConnectionPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/serviceConnectionPolicies', request_field='', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionPoliciesListRequest', response_type_name='ListServiceConnectionPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single ServiceConnectionPolicy.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionPolicies/{serviceConnectionPoliciesId}', http_method='PATCH', method_id='networkconnectivity.projects.locations.serviceConnectionPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='serviceConnectionPolicy', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionPoliciesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionPoliciesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionPolicies/{serviceConnectionPoliciesId}:setIamPolicy', http_method='POST', method_id='networkconnectivity.projects.locations.serviceConnectionPolicies.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionPoliciesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionPolicies/{serviceConnectionPoliciesId}:testIamPermissions', http_method='POST', method_id='networkconnectivity.projects.locations.serviceConnectionPolicies.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionPoliciesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)