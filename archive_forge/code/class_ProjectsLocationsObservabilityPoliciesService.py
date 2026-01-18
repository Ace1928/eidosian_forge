from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1alpha1 import networkservices_v1alpha1_messages as messages
class ProjectsLocationsObservabilityPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_observabilityPolicies resource."""
    _NAME = 'projects_locations_observabilityPolicies'

    def __init__(self, client):
        super(NetworkservicesV1alpha1.ProjectsLocationsObservabilityPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ObservabilityPolicy in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsObservabilityPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/observabilityPolicies', http_method='POST', method_id='networkservices.projects.locations.observabilityPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['observabilityPolicyId'], relative_path='v1alpha1/{+parent}/observabilityPolicies', request_field='observabilityPolicy', request_type_name='NetworkservicesProjectsLocationsObservabilityPoliciesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ObservabilityPolicy.

      Args:
        request: (NetworkservicesProjectsLocationsObservabilityPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/observabilityPolicies/{observabilityPoliciesId}', http_method='DELETE', method_id='networkservices.projects.locations.observabilityPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsObservabilityPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ObservabilityPolicy.

      Args:
        request: (NetworkservicesProjectsLocationsObservabilityPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ObservabilityPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/observabilityPolicies/{observabilityPoliciesId}', http_method='GET', method_id='networkservices.projects.locations.observabilityPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsObservabilityPoliciesGetRequest', response_type_name='ObservabilityPolicy', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworkservicesProjectsLocationsObservabilityPoliciesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/observabilityPolicies/{observabilityPoliciesId}:getIamPolicy', http_method='GET', method_id='networkservices.projects.locations.observabilityPolicies.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha1/{+resource}:getIamPolicy', request_field='', request_type_name='NetworkservicesProjectsLocationsObservabilityPoliciesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ObservabilityPolicies in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsObservabilityPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListObservabilityPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/observabilityPolicies', http_method='GET', method_id='networkservices.projects.locations.observabilityPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/observabilityPolicies', request_field='', request_type_name='NetworkservicesProjectsLocationsObservabilityPoliciesListRequest', response_type_name='ListObservabilityPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single ObservabilityPolicy.

      Args:
        request: (NetworkservicesProjectsLocationsObservabilityPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/observabilityPolicies/{observabilityPoliciesId}', http_method='PATCH', method_id='networkservices.projects.locations.observabilityPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='observabilityPolicy', request_type_name='NetworkservicesProjectsLocationsObservabilityPoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworkservicesProjectsLocationsObservabilityPoliciesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/observabilityPolicies/{observabilityPoliciesId}:setIamPolicy', http_method='POST', method_id='networkservices.projects.locations.observabilityPolicies.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='NetworkservicesProjectsLocationsObservabilityPoliciesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworkservicesProjectsLocationsObservabilityPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/observabilityPolicies/{observabilityPoliciesId}:testIamPermissions', http_method='POST', method_id='networkservices.projects.locations.observabilityPolicies.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='NetworkservicesProjectsLocationsObservabilityPoliciesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)