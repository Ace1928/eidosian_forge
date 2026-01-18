from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsModelsEvaluationsSlicesService(base_api.BaseApiService):
    """Service class for the projects_locations_models_evaluations_slices resource."""
    _NAME = 'projects_locations_models_evaluations_slices'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsModelsEvaluationsSlicesService, self).__init__(client)
        self._upload_configs = {}

    def BatchImport(self, request, global_params=None):
        """Imports a list of externally generated EvaluatedAnnotations.

      Args:
        request: (AiplatformProjectsLocationsModelsEvaluationsSlicesBatchImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1BatchImportEvaluatedAnnotationsResponse) The response message.
      """
        config = self.GetMethodConfig('BatchImport')
        return self._RunMethod(config, request, global_params=global_params)
    BatchImport.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}/evaluations/{evaluationsId}/slices/{slicesId}:batchImport', http_method='POST', method_id='aiplatform.projects.locations.models.evaluations.slices.batchImport', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:batchImport', request_field='googleCloudAiplatformV1BatchImportEvaluatedAnnotationsRequest', request_type_name='AiplatformProjectsLocationsModelsEvaluationsSlicesBatchImportRequest', response_type_name='GoogleCloudAiplatformV1BatchImportEvaluatedAnnotationsResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a ModelEvaluationSlice.

      Args:
        request: (AiplatformProjectsLocationsModelsEvaluationsSlicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ModelEvaluationSlice) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}/evaluations/{evaluationsId}/slices/{slicesId}', http_method='GET', method_id='aiplatform.projects.locations.models.evaluations.slices.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsModelsEvaluationsSlicesGetRequest', response_type_name='GoogleCloudAiplatformV1ModelEvaluationSlice', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ModelEvaluationSlices in a ModelEvaluation.

      Args:
        request: (AiplatformProjectsLocationsModelsEvaluationsSlicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListModelEvaluationSlicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}/evaluations/{evaluationsId}/slices', http_method='GET', method_id='aiplatform.projects.locations.models.evaluations.slices.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/slices', request_field='', request_type_name='AiplatformProjectsLocationsModelsEvaluationsSlicesListRequest', response_type_name='GoogleCloudAiplatformV1ListModelEvaluationSlicesResponse', supports_download=False)