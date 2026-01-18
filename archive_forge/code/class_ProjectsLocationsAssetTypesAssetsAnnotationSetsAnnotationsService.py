from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.mediaasset.v1alpha import mediaasset_v1alpha_messages as messages
class ProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsService(base_api.BaseApiService):
    """Service class for the projects_locations_assetTypes_assets_annotationSets_annotations resource."""
    _NAME = 'projects_locations_assetTypes_assets_annotationSets_annotations'

    def __init__(self, client):
        super(MediaassetV1alpha.ProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new annotation in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Annotation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}/annotations', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.annotations.create', ordered_params=['parent'], path_params=['parent'], query_params=['annotationId'], relative_path='v1alpha/{+parent}/annotations', request_field='annotation', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsCreateRequest', response_type_name='Annotation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single annotation.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}/annotations/{annotationsId}', http_method='DELETE', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.annotations.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single annotation.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Annotation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}/annotations/{annotationsId}', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.annotations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsGetRequest', response_type_name='Annotation', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}/annotations/{annotationsId}:getIamPolicy', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.annotations.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha/{+resource}:getIamPolicy', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists annotations in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAnnotationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}/annotations', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.annotations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/annotations', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsListRequest', response_type_name='ListAnnotationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single annotation.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Annotation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}/annotations/{annotationsId}', http_method='PATCH', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.annotations.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha/{+name}', request_field='annotation', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsPatchRequest', response_type_name='Annotation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}/annotations/{annotationsId}:setIamPolicy', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.annotations.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}/annotations/{annotationsId}:testIamPermissions', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.annotations.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsAnnotationsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)