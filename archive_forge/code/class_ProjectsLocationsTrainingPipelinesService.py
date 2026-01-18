from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsTrainingPipelinesService(base_api.BaseApiService):
    """Service class for the projects_locations_trainingPipelines resource."""
    _NAME = 'projects_locations_trainingPipelines'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsTrainingPipelinesService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels a TrainingPipeline. Starts asynchronous cancellation on the TrainingPipeline. The server makes a best effort to cancel the pipeline, but success is not guaranteed. Clients can use PipelineService.GetTrainingPipeline or other methods to check whether the cancellation succeeded or whether the pipeline completed despite cancellation. On successful cancellation, the TrainingPipeline is not deleted; instead it becomes a pipeline with a TrainingPipeline.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`, and TrainingPipeline.state is set to `CANCELLED`.

      Args:
        request: (AiplatformProjectsLocationsTrainingPipelinesCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/trainingPipelines/{trainingPipelinesId}:cancel', http_method='POST', method_id='aiplatform.projects.locations.trainingPipelines.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='googleCloudAiplatformV1CancelTrainingPipelineRequest', request_type_name='AiplatformProjectsLocationsTrainingPipelinesCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a TrainingPipeline. A created TrainingPipeline right away will be attempted to be run.

      Args:
        request: (AiplatformProjectsLocationsTrainingPipelinesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1TrainingPipeline) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/trainingPipelines', http_method='POST', method_id='aiplatform.projects.locations.trainingPipelines.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/trainingPipelines', request_field='googleCloudAiplatformV1TrainingPipeline', request_type_name='AiplatformProjectsLocationsTrainingPipelinesCreateRequest', response_type_name='GoogleCloudAiplatformV1TrainingPipeline', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a TrainingPipeline.

      Args:
        request: (AiplatformProjectsLocationsTrainingPipelinesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/trainingPipelines/{trainingPipelinesId}', http_method='DELETE', method_id='aiplatform.projects.locations.trainingPipelines.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsTrainingPipelinesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a TrainingPipeline.

      Args:
        request: (AiplatformProjectsLocationsTrainingPipelinesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1TrainingPipeline) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/trainingPipelines/{trainingPipelinesId}', http_method='GET', method_id='aiplatform.projects.locations.trainingPipelines.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsTrainingPipelinesGetRequest', response_type_name='GoogleCloudAiplatformV1TrainingPipeline', supports_download=False)

    def List(self, request, global_params=None):
        """Lists TrainingPipelines in a Location.

      Args:
        request: (AiplatformProjectsLocationsTrainingPipelinesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListTrainingPipelinesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/trainingPipelines', http_method='GET', method_id='aiplatform.projects.locations.trainingPipelines.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/trainingPipelines', request_field='', request_type_name='AiplatformProjectsLocationsTrainingPipelinesListRequest', response_type_name='GoogleCloudAiplatformV1ListTrainingPipelinesResponse', supports_download=False)