from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsServiceLbPoliciesService(base_api.BaseApiService):
    """Service class for the projects_locations_serviceLbPolicies resource."""
    _NAME = 'projects_locations_serviceLbPolicies'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsServiceLbPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ServiceLbPolicy in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsServiceLbPoliciesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceLbPolicies', http_method='POST', method_id='networkservices.projects.locations.serviceLbPolicies.create', ordered_params=['parent'], path_params=['parent'], query_params=['serviceLbPolicyId'], relative_path='v1/{+parent}/serviceLbPolicies', request_field='serviceLbPolicy', request_type_name='NetworkservicesProjectsLocationsServiceLbPoliciesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ServiceLbPolicy.

      Args:
        request: (NetworkservicesProjectsLocationsServiceLbPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceLbPolicies/{serviceLbPoliciesId}', http_method='DELETE', method_id='networkservices.projects.locations.serviceLbPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsServiceLbPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ServiceLbPolicy.

      Args:
        request: (NetworkservicesProjectsLocationsServiceLbPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceLbPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceLbPolicies/{serviceLbPoliciesId}', http_method='GET', method_id='networkservices.projects.locations.serviceLbPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsServiceLbPoliciesGetRequest', response_type_name='ServiceLbPolicy', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworkservicesProjectsLocationsServiceLbPoliciesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceLbPolicies/{serviceLbPoliciesId}:getIamPolicy', http_method='GET', method_id='networkservices.projects.locations.serviceLbPolicies.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='NetworkservicesProjectsLocationsServiceLbPoliciesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ServiceLbPolicies in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsServiceLbPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceLbPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceLbPolicies', http_method='GET', method_id='networkservices.projects.locations.serviceLbPolicies.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/serviceLbPolicies', request_field='', request_type_name='NetworkservicesProjectsLocationsServiceLbPoliciesListRequest', response_type_name='ListServiceLbPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single ServiceLbPolicy.

      Args:
        request: (NetworkservicesProjectsLocationsServiceLbPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceLbPolicies/{serviceLbPoliciesId}', http_method='PATCH', method_id='networkservices.projects.locations.serviceLbPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='serviceLbPolicy', request_type_name='NetworkservicesProjectsLocationsServiceLbPoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworkservicesProjectsLocationsServiceLbPoliciesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceLbPolicies/{serviceLbPoliciesId}:setIamPolicy', http_method='POST', method_id='networkservices.projects.locations.serviceLbPolicies.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='NetworkservicesProjectsLocationsServiceLbPoliciesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworkservicesProjectsLocationsServiceLbPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceLbPolicies/{serviceLbPoliciesId}:testIamPermissions', http_method='POST', method_id='networkservices.projects.locations.serviceLbPolicies.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='NetworkservicesProjectsLocationsServiceLbPoliciesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)