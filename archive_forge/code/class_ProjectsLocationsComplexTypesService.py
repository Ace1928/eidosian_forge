from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.mediaasset.v1alpha import mediaasset_v1alpha_messages as messages
class ProjectsLocationsComplexTypesService(base_api.BaseApiService):
    """Service class for the projects_locations_complexTypes resource."""
    _NAME = 'projects_locations_complexTypes'

    def __init__(self, client):
        super(MediaassetV1alpha.ProjectsLocationsComplexTypesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new complex type in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsComplexTypesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/complexTypes', http_method='POST', method_id='mediaasset.projects.locations.complexTypes.create', ordered_params=['parent'], path_params=['parent'], query_params=['complexTypeId', 'requestId'], relative_path='v1alpha/{+parent}/complexTypes', request_field='complexType', request_type_name='MediaassetProjectsLocationsComplexTypesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single complex type.

      Args:
        request: (MediaassetProjectsLocationsComplexTypesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/complexTypes/{complexTypesId}', http_method='DELETE', method_id='mediaasset.projects.locations.complexTypes.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsComplexTypesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single complex type.

      Args:
        request: (MediaassetProjectsLocationsComplexTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComplexType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/complexTypes/{complexTypesId}', http_method='GET', method_id='mediaasset.projects.locations.complexTypes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsComplexTypesGetRequest', response_type_name='ComplexType', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (MediaassetProjectsLocationsComplexTypesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/complexTypes/{complexTypesId}:getIamPolicy', http_method='GET', method_id='mediaasset.projects.locations.complexTypes.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha/{+resource}:getIamPolicy', request_field='', request_type_name='MediaassetProjectsLocationsComplexTypesGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists complex types in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsComplexTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListComplexTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/complexTypes', http_method='GET', method_id='mediaasset.projects.locations.complexTypes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/complexTypes', request_field='', request_type_name='MediaassetProjectsLocationsComplexTypesListRequest', response_type_name='ListComplexTypesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single complex type.

      Args:
        request: (MediaassetProjectsLocationsComplexTypesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/complexTypes/{complexTypesId}', http_method='PATCH', method_id='mediaasset.projects.locations.complexTypes.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='complexType', request_type_name='MediaassetProjectsLocationsComplexTypesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (MediaassetProjectsLocationsComplexTypesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/complexTypes/{complexTypesId}:setIamPolicy', http_method='POST', method_id='mediaasset.projects.locations.complexTypes.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='MediaassetProjectsLocationsComplexTypesSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (MediaassetProjectsLocationsComplexTypesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/complexTypes/{complexTypesId}:testIamPermissions', http_method='POST', method_id='mediaasset.projects.locations.complexTypes.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='MediaassetProjectsLocationsComplexTypesTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)