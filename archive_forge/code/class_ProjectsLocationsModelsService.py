from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsModelsService(base_api.BaseApiService):
    """Service class for the projects_locations_models resource."""
    _NAME = 'projects_locations_models'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsModelsService, self).__init__(client)
        self._upload_configs = {}

    def Copy(self, request, global_params=None):
        """Copies an already existing Vertex AI Model into the specified Location. The source Model must exist in the same Project. When copying custom Models, the users themselves are responsible for Model.metadata content to be region-agnostic, as well as making sure that any resources (e.g. files) it depends on remain accessible.

      Args:
        request: (AiplatformProjectsLocationsModelsCopyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Copy')
        return self._RunMethod(config, request, global_params=global_params)
    Copy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models:copy', http_method='POST', method_id='aiplatform.projects.locations.models.copy', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/models:copy', request_field='googleCloudAiplatformV1CopyModelRequest', request_type_name='AiplatformProjectsLocationsModelsCopyRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Model. A model cannot be deleted if any Endpoint resource has a DeployedModel based on the model in its deployed_models field.

      Args:
        request: (AiplatformProjectsLocationsModelsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}', http_method='DELETE', method_id='aiplatform.projects.locations.models.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsModelsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def DeleteVersion(self, request, global_params=None):
        """Deletes a Model version. Model version can only be deleted if there are no DeployedModels created from it. Deleting the only version in the Model is not allowed. Use DeleteModel for deleting the Model instead.

      Args:
        request: (AiplatformProjectsLocationsModelsDeleteVersionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('DeleteVersion')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteVersion.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}:deleteVersion', http_method='DELETE', method_id='aiplatform.projects.locations.models.deleteVersion', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:deleteVersion', request_field='', request_type_name='AiplatformProjectsLocationsModelsDeleteVersionRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Export(self, request, global_params=None):
        """Exports a trained, exportable Model to a location specified by the user. A Model is considered to be exportable if it has at least one supported export format.

      Args:
        request: (AiplatformProjectsLocationsModelsExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}:export', http_method='POST', method_id='aiplatform.projects.locations.models.export', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:export', request_field='googleCloudAiplatformV1ExportModelRequest', request_type_name='AiplatformProjectsLocationsModelsExportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Model.

      Args:
        request: (AiplatformProjectsLocationsModelsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Model) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}', http_method='GET', method_id='aiplatform.projects.locations.models.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsModelsGetRequest', response_type_name='GoogleCloudAiplatformV1Model', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (AiplatformProjectsLocationsModelsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}:getIamPolicy', http_method='POST', method_id='aiplatform.projects.locations.models.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='AiplatformProjectsLocationsModelsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Models in a Location.

      Args:
        request: (AiplatformProjectsLocationsModelsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListModelsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models', http_method='GET', method_id='aiplatform.projects.locations.models.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/models', request_field='', request_type_name='AiplatformProjectsLocationsModelsListRequest', response_type_name='GoogleCloudAiplatformV1ListModelsResponse', supports_download=False)

    def ListVersions(self, request, global_params=None):
        """Lists versions of the specified model.

      Args:
        request: (AiplatformProjectsLocationsModelsListVersionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListModelVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('ListVersions')
        return self._RunMethod(config, request, global_params=global_params)
    ListVersions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}:listVersions', http_method='GET', method_id='aiplatform.projects.locations.models.listVersions', ordered_params=['name'], path_params=['name'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+name}:listVersions', request_field='', request_type_name='AiplatformProjectsLocationsModelsListVersionsRequest', response_type_name='GoogleCloudAiplatformV1ListModelVersionsResponse', supports_download=False)

    def MergeVersionAliases(self, request, global_params=None):
        """Merges a set of aliases for a Model version.

      Args:
        request: (AiplatformProjectsLocationsModelsMergeVersionAliasesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Model) The response message.
      """
        config = self.GetMethodConfig('MergeVersionAliases')
        return self._RunMethod(config, request, global_params=global_params)
    MergeVersionAliases.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}:mergeVersionAliases', http_method='POST', method_id='aiplatform.projects.locations.models.mergeVersionAliases', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:mergeVersionAliases', request_field='googleCloudAiplatformV1MergeVersionAliasesRequest', request_type_name='AiplatformProjectsLocationsModelsMergeVersionAliasesRequest', response_type_name='GoogleCloudAiplatformV1Model', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Model.

      Args:
        request: (AiplatformProjectsLocationsModelsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Model) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}', http_method='PATCH', method_id='aiplatform.projects.locations.models.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1Model', request_type_name='AiplatformProjectsLocationsModelsPatchRequest', response_type_name='GoogleCloudAiplatformV1Model', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (AiplatformProjectsLocationsModelsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}:setIamPolicy', http_method='POST', method_id='aiplatform.projects.locations.models.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='AiplatformProjectsLocationsModelsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (AiplatformProjectsLocationsModelsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}:testIamPermissions', http_method='POST', method_id='aiplatform.projects.locations.models.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=['permissions'], relative_path='v1/{+resource}:testIamPermissions', request_field='', request_type_name='AiplatformProjectsLocationsModelsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)

    def UpdateExplanationDataset(self, request, global_params=None):
        """Incrementally update the dataset used for an examples model.

      Args:
        request: (AiplatformProjectsLocationsModelsUpdateExplanationDatasetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('UpdateExplanationDataset')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateExplanationDataset.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}:updateExplanationDataset', http_method='POST', method_id='aiplatform.projects.locations.models.updateExplanationDataset', ordered_params=['model'], path_params=['model'], query_params=[], relative_path='v1/{+model}:updateExplanationDataset', request_field='googleCloudAiplatformV1UpdateExplanationDatasetRequest', request_type_name='AiplatformProjectsLocationsModelsUpdateExplanationDatasetRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Upload(self, request, global_params=None):
        """Uploads a Model artifact into Vertex AI.

      Args:
        request: (AiplatformProjectsLocationsModelsUploadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Upload')
        return self._RunMethod(config, request, global_params=global_params)
    Upload.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models:upload', http_method='POST', method_id='aiplatform.projects.locations.models.upload', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/models:upload', request_field='googleCloudAiplatformV1UploadModelRequest', request_type_name='AiplatformProjectsLocationsModelsUploadRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)