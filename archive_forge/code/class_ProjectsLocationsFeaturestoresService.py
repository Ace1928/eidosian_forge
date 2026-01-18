from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsFeaturestoresService(base_api.BaseApiService):
    """Service class for the projects_locations_featurestores resource."""
    _NAME = 'projects_locations_featurestores'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsFeaturestoresService, self).__init__(client)
        self._upload_configs = {}

    def BatchReadFeatureValues(self, request, global_params=None):
        """Batch reads Feature values from a Featurestore. This API enables batch reading Feature values, where each read instance in the batch may read Feature values of entities from one or more EntityTypes. Point-in-time correctness is guaranteed for Feature values of each read instance as of each instance's read timestamp.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresBatchReadFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchReadFeatureValues')
        return self._RunMethod(config, request, global_params=global_params)
    BatchReadFeatureValues.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}:batchReadFeatureValues', http_method='POST', method_id='aiplatform.projects.locations.featurestores.batchReadFeatureValues', ordered_params=['featurestore'], path_params=['featurestore'], query_params=[], relative_path='v1/{+featurestore}:batchReadFeatureValues', request_field='googleCloudAiplatformV1BatchReadFeatureValuesRequest', request_type_name='AiplatformProjectsLocationsFeaturestoresBatchReadFeatureValuesRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new Featurestore in a given project and location.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores', http_method='POST', method_id='aiplatform.projects.locations.featurestores.create', ordered_params=['parent'], path_params=['parent'], query_params=['featurestoreId'], relative_path='v1/{+parent}/featurestores', request_field='googleCloudAiplatformV1Featurestore', request_type_name='AiplatformProjectsLocationsFeaturestoresCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Featurestore. The Featurestore must not contain any EntityTypes or `force` must be set to true for the request to succeed.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}', http_method='DELETE', method_id='aiplatform.projects.locations.featurestores.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Featurestore.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Featurestore) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}', http_method='GET', method_id='aiplatform.projects.locations.featurestores.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresGetRequest', response_type_name='GoogleCloudAiplatformV1Featurestore', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}:getIamPolicy', http_method='POST', method_id='aiplatform.projects.locations.featurestores.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Featurestores in a given project and location.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListFeaturestoresResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores', http_method='GET', method_id='aiplatform.projects.locations.featurestores.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/featurestores', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresListRequest', response_type_name='GoogleCloudAiplatformV1ListFeaturestoresResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Featurestore.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}', http_method='PATCH', method_id='aiplatform.projects.locations.featurestores.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1Featurestore', request_type_name='AiplatformProjectsLocationsFeaturestoresPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SearchFeatures(self, request, global_params=None):
        """Searches Features matching a query in a given project.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresSearchFeaturesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1SearchFeaturesResponse) The response message.
      """
        config = self.GetMethodConfig('SearchFeatures')
        return self._RunMethod(config, request, global_params=global_params)
    SearchFeatures.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores:searchFeatures', http_method='GET', method_id='aiplatform.projects.locations.featurestores.searchFeatures', ordered_params=['location'], path_params=['location'], query_params=['pageSize', 'pageToken', 'query'], relative_path='v1/{+location}/featurestores:searchFeatures', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresSearchFeaturesRequest', response_type_name='GoogleCloudAiplatformV1SearchFeaturesResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}:setIamPolicy', http_method='POST', method_id='aiplatform.projects.locations.featurestores.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='AiplatformProjectsLocationsFeaturestoresSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}:testIamPermissions', http_method='POST', method_id='aiplatform.projects.locations.featurestores.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=['permissions'], relative_path='v1/{+resource}:testIamPermissions', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)