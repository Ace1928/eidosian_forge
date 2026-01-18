from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsModelsEvaluationsService(base_api.BaseApiService):
    """Service class for the projects_locations_models_evaluations resource."""
    _NAME = 'projects_locations_models_evaluations'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsModelsEvaluationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a ModelEvaluation.

      Args:
        request: (AiplatformProjectsLocationsModelsEvaluationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ModelEvaluation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}/evaluations/{evaluationsId}', http_method='GET', method_id='aiplatform.projects.locations.models.evaluations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsModelsEvaluationsGetRequest', response_type_name='GoogleCloudAiplatformV1ModelEvaluation', supports_download=False)

    def Import(self, request, global_params=None):
        """Imports an externally generated ModelEvaluation.

      Args:
        request: (AiplatformProjectsLocationsModelsEvaluationsImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ModelEvaluation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}/evaluations:import', http_method='POST', method_id='aiplatform.projects.locations.models.evaluations.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/evaluations:import', request_field='googleCloudAiplatformV1ImportModelEvaluationRequest', request_type_name='AiplatformProjectsLocationsModelsEvaluationsImportRequest', response_type_name='GoogleCloudAiplatformV1ModelEvaluation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ModelEvaluations in a Model.

      Args:
        request: (AiplatformProjectsLocationsModelsEvaluationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListModelEvaluationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/models/{modelsId}/evaluations', http_method='GET', method_id='aiplatform.projects.locations.models.evaluations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/evaluations', request_field='', request_type_name='AiplatformProjectsLocationsModelsEvaluationsListRequest', response_type_name='GoogleCloudAiplatformV1ListModelEvaluationsResponse', supports_download=False)