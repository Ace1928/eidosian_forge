from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.mediaasset.v1alpha import mediaasset_v1alpha_messages as messages
class ProjectsLocationsAssetTypesService(base_api.BaseApiService):
    """Service class for the projects_locations_assetTypes resource."""
    _NAME = 'projects_locations_assetTypes'

    def __init__(self, client):
        super(MediaassetV1alpha.ProjectsLocationsAssetTypesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new AssetType in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.create', ordered_params=['parent'], path_params=['parent'], query_params=['assetTypeId', 'requestId'], relative_path='v1alpha/{+parent}/assetTypes', request_field='assetType', request_type_name='MediaassetProjectsLocationsAssetTypesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single AssetType.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}', http_method='DELETE', method_id='mediaasset.projects.locations.assetTypes.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single AssetType.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AssetType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesGetRequest', response_type_name='AssetType', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}:getIamPolicy', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha/{+resource}:getIamPolicy', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists AssetTypes in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAssetTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/assetTypes', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesListRequest', response_type_name='ListAssetTypesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single AssetType.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}', http_method='PATCH', method_id='mediaasset.projects.locations.assetTypes.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='assetType', request_type_name='MediaassetProjectsLocationsAssetTypesPatchRequest', response_type_name='Operation', supports_download=False)

    def Search(self, request, global_params=None):
        """Search returns the resources (e.g., assets and annotations) under a Video Asset Type that match the given query. Search covers both media content and metadata.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchAssetTypeResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}:search', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.search', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:search', request_field='searchAssetTypeRequest', request_type_name='MediaassetProjectsLocationsAssetTypesSearchRequest', response_type_name='SearchAssetTypeResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}:setIamPolicy', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='MediaassetProjectsLocationsAssetTypesSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}:testIamPermissions', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='MediaassetProjectsLocationsAssetTypesTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)