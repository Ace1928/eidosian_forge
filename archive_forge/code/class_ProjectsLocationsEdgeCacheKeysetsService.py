from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsEdgeCacheKeysetsService(base_api.BaseApiService):
    """Service class for the projects_locations_edgeCacheKeysets resource."""
    _NAME = 'projects_locations_edgeCacheKeysets'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsEdgeCacheKeysetsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new EdgeCacheKeyset in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsEdgeCacheKeysetsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeCacheKeysets', http_method='POST', method_id='networkservices.projects.locations.edgeCacheKeysets.create', ordered_params=['parent'], path_params=['parent'], query_params=['edgeCacheKeysetId'], relative_path='v1/{+parent}/edgeCacheKeysets', request_field='edgeCacheKeyset', request_type_name='NetworkservicesProjectsLocationsEdgeCacheKeysetsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single EdgeCacheKeyset.

      Args:
        request: (NetworkservicesProjectsLocationsEdgeCacheKeysetsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeCacheKeysets/{edgeCacheKeysetsId}', http_method='DELETE', method_id='networkservices.projects.locations.edgeCacheKeysets.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsEdgeCacheKeysetsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single EdgeCacheKeyset.

      Args:
        request: (NetworkservicesProjectsLocationsEdgeCacheKeysetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EdgeCacheKeyset) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeCacheKeysets/{edgeCacheKeysetsId}', http_method='GET', method_id='networkservices.projects.locations.edgeCacheKeysets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsEdgeCacheKeysetsGetRequest', response_type_name='EdgeCacheKeyset', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworkservicesProjectsLocationsEdgeCacheKeysetsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeCacheKeysets/{edgeCacheKeysetsId}:getIamPolicy', http_method='GET', method_id='networkservices.projects.locations.edgeCacheKeysets.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='NetworkservicesProjectsLocationsEdgeCacheKeysetsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists EdgeCacheKeysets in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsEdgeCacheKeysetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEdgeCacheKeysetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeCacheKeysets', http_method='GET', method_id='networkservices.projects.locations.edgeCacheKeysets.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/edgeCacheKeysets', request_field='', request_type_name='NetworkservicesProjectsLocationsEdgeCacheKeysetsListRequest', response_type_name='ListEdgeCacheKeysetsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single EdgeCacheKeyset.

      Args:
        request: (NetworkservicesProjectsLocationsEdgeCacheKeysetsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeCacheKeysets/{edgeCacheKeysetsId}', http_method='PATCH', method_id='networkservices.projects.locations.edgeCacheKeysets.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='edgeCacheKeyset', request_type_name='NetworkservicesProjectsLocationsEdgeCacheKeysetsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworkservicesProjectsLocationsEdgeCacheKeysetsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeCacheKeysets/{edgeCacheKeysetsId}:setIamPolicy', http_method='POST', method_id='networkservices.projects.locations.edgeCacheKeysets.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='NetworkservicesProjectsLocationsEdgeCacheKeysetsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworkservicesProjectsLocationsEdgeCacheKeysetsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeCacheKeysets/{edgeCacheKeysetsId}:testIamPermissions', http_method='POST', method_id='networkservices.projects.locations.edgeCacheKeysets.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='NetworkservicesProjectsLocationsEdgeCacheKeysetsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)