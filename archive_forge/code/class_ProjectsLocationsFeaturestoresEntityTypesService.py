from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsFeaturestoresEntityTypesService(base_api.BaseApiService):
    """Service class for the projects_locations_featurestores_entityTypes resource."""
    _NAME = 'projects_locations_featurestores_entityTypes'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsFeaturestoresEntityTypesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new EntityType in a given Featurestore.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.create', ordered_params=['parent'], path_params=['parent'], query_params=['entityTypeId'], relative_path='v1/{+parent}/entityTypes', request_field='googleCloudAiplatformV1EntityType', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single EntityType. The EntityType must not have any Features or `force` must be set to true for the request to succeed.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}', http_method='DELETE', method_id='aiplatform.projects.locations.featurestores.entityTypes.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def DeleteFeatureValues(self, request, global_params=None):
        """Delete Feature values from Featurestore. The progress of the deletion is tracked by the returned operation. The deleted feature values are guaranteed to be invisible to subsequent read operations after the operation is marked as successfully done. If a delete feature values operation fails, the feature values returned from reads and exports may be inconsistent. If consistency is required, the caller must retry the same delete request again and wait till the new operation returned is marked as successfully done.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesDeleteFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('DeleteFeatureValues')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteFeatureValues.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}:deleteFeatureValues', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.deleteFeatureValues', ordered_params=['entityType'], path_params=['entityType'], query_params=[], relative_path='v1/{+entityType}:deleteFeatureValues', request_field='googleCloudAiplatformV1DeleteFeatureValuesRequest', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesDeleteFeatureValuesRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def ExportFeatureValues(self, request, global_params=None):
        """Exports Feature values from all the entities of a target EntityType.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesExportFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('ExportFeatureValues')
        return self._RunMethod(config, request, global_params=global_params)
    ExportFeatureValues.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}:exportFeatureValues', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.exportFeatureValues', ordered_params=['entityType'], path_params=['entityType'], query_params=[], relative_path='v1/{+entityType}:exportFeatureValues', request_field='googleCloudAiplatformV1ExportFeatureValuesRequest', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesExportFeatureValuesRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single EntityType.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1EntityType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}', http_method='GET', method_id='aiplatform.projects.locations.featurestores.entityTypes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesGetRequest', response_type_name='GoogleCloudAiplatformV1EntityType', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}:getIamPolicy', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def ImportFeatureValues(self, request, global_params=None):
        """Imports Feature values into the Featurestore from a source storage. The progress of the import is tracked by the returned operation. The imported features are guaranteed to be visible to subsequent read operations after the operation is marked as successfully done. If an import operation fails, the Feature values returned from reads and exports may be inconsistent. If consistency is required, the caller must retry the same import request again and wait till the new operation returned is marked as successfully done. There are also scenarios where the caller can cause inconsistency. - Source data for import contains multiple distinct Feature values for the same entity ID and timestamp. - Source is modified during an import. This includes adding, updating, or removing source data and/or metadata. Examples of updating metadata include but are not limited to changing storage location, storage class, or retention policy. - Online serving cluster is under-provisioned.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesImportFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('ImportFeatureValues')
        return self._RunMethod(config, request, global_params=global_params)
    ImportFeatureValues.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}:importFeatureValues', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.importFeatureValues', ordered_params=['entityType'], path_params=['entityType'], query_params=[], relative_path='v1/{+entityType}:importFeatureValues', request_field='googleCloudAiplatformV1ImportFeatureValuesRequest', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesImportFeatureValuesRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists EntityTypes in a given Featurestore.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListEntityTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes', http_method='GET', method_id='aiplatform.projects.locations.featurestores.entityTypes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/entityTypes', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesListRequest', response_type_name='GoogleCloudAiplatformV1ListEntityTypesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single EntityType.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1EntityType) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}', http_method='PATCH', method_id='aiplatform.projects.locations.featurestores.entityTypes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1EntityType', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesPatchRequest', response_type_name='GoogleCloudAiplatformV1EntityType', supports_download=False)

    def ReadFeatureValues(self, request, global_params=None):
        """Reads Feature values of a specific entity of an EntityType. For reading feature values of multiple entities of an EntityType, please use StreamingReadFeatureValues.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesReadFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ReadFeatureValuesResponse) The response message.
      """
        config = self.GetMethodConfig('ReadFeatureValues')
        return self._RunMethod(config, request, global_params=global_params)
    ReadFeatureValues.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}:readFeatureValues', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.readFeatureValues', ordered_params=['entityType'], path_params=['entityType'], query_params=[], relative_path='v1/{+entityType}:readFeatureValues', request_field='googleCloudAiplatformV1ReadFeatureValuesRequest', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesReadFeatureValuesRequest', response_type_name='GoogleCloudAiplatformV1ReadFeatureValuesResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}:setIamPolicy', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def StreamingReadFeatureValues(self, request, global_params=None):
        """Reads Feature values for multiple entities. Depending on their size, data for different entities may be broken up across multiple responses.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesStreamingReadFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ReadFeatureValuesResponse) The response message.
      """
        config = self.GetMethodConfig('StreamingReadFeatureValues')
        return self._RunMethod(config, request, global_params=global_params)
    StreamingReadFeatureValues.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}:streamingReadFeatureValues', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.streamingReadFeatureValues', ordered_params=['entityType'], path_params=['entityType'], query_params=[], relative_path='v1/{+entityType}:streamingReadFeatureValues', request_field='googleCloudAiplatformV1StreamingReadFeatureValuesRequest', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesStreamingReadFeatureValuesRequest', response_type_name='GoogleCloudAiplatformV1ReadFeatureValuesResponse', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}:testIamPermissions', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=['permissions'], relative_path='v1/{+resource}:testIamPermissions', request_field='', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)

    def WriteFeatureValues(self, request, global_params=None):
        """Writes Feature values of one or more entities of an EntityType. The Feature values are merged into existing entities if any. The Feature values to be written must have timestamp within the online storage retention.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesWriteFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1WriteFeatureValuesResponse) The response message.
      """
        config = self.GetMethodConfig('WriteFeatureValues')
        return self._RunMethod(config, request, global_params=global_params)
    WriteFeatureValues.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/featurestores/{featurestoresId}/entityTypes/{entityTypesId}:writeFeatureValues', http_method='POST', method_id='aiplatform.projects.locations.featurestores.entityTypes.writeFeatureValues', ordered_params=['entityType'], path_params=['entityType'], query_params=[], relative_path='v1/{+entityType}:writeFeatureValues', request_field='googleCloudAiplatformV1WriteFeatureValuesRequest', request_type_name='AiplatformProjectsLocationsFeaturestoresEntityTypesWriteFeatureValuesRequest', response_type_name='GoogleCloudAiplatformV1WriteFeatureValuesResponse', supports_download=False)