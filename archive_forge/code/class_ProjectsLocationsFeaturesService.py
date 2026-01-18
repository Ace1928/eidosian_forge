from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
class ProjectsLocationsFeaturesService(base_api.BaseApiService):
    """Service class for the projects_locations_features resource."""
    _NAME = 'projects_locations_features'

    def __init__(self, client):
        super(GkehubV1beta.ProjectsLocationsFeaturesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Adds a new Feature.

      Args:
        request: (GkehubProjectsLocationsFeaturesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/features', http_method='POST', method_id='gkehub.projects.locations.features.create', ordered_params=['parent'], path_params=['parent'], query_params=['featureId', 'requestId'], relative_path='v1beta/{+parent}/features', request_field='feature', request_type_name='GkehubProjectsLocationsFeaturesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Removes a Feature.

      Args:
        request: (GkehubProjectsLocationsFeaturesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/features/{featuresId}', http_method='DELETE', method_id='gkehub.projects.locations.features.delete', ordered_params=['name'], path_params=['name'], query_params=['force', 'requestId'], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsFeaturesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Feature.

      Args:
        request: (GkehubProjectsLocationsFeaturesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Feature) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/features/{featuresId}', http_method='GET', method_id='gkehub.projects.locations.features.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsFeaturesGetRequest', response_type_name='Feature', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (GkehubProjectsLocationsFeaturesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/features/{featuresId}:getIamPolicy', http_method='GET', method_id='gkehub.projects.locations.features.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1beta/{+resource}:getIamPolicy', request_field='', request_type_name='GkehubProjectsLocationsFeaturesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Features in a given project and location.

      Args:
        request: (GkehubProjectsLocationsFeaturesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFeaturesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/features', http_method='GET', method_id='gkehub.projects.locations.features.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/features', request_field='', request_type_name='GkehubProjectsLocationsFeaturesListRequest', response_type_name='ListFeaturesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing Feature.

      Args:
        request: (GkehubProjectsLocationsFeaturesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/features/{featuresId}', http_method='PATCH', method_id='gkehub.projects.locations.features.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1beta/{+name}', request_field='feature', request_type_name='GkehubProjectsLocationsFeaturesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (GkehubProjectsLocationsFeaturesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/features/{featuresId}:setIamPolicy', http_method='POST', method_id='gkehub.projects.locations.features.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='GkehubProjectsLocationsFeaturesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (GkehubProjectsLocationsFeaturesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/features/{featuresId}:testIamPermissions', http_method='POST', method_id='gkehub.projects.locations.features.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='GkehubProjectsLocationsFeaturesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)