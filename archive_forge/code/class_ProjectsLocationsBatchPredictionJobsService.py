from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsBatchPredictionJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_batchPredictionJobs resource."""
    _NAME = 'projects_locations_batchPredictionJobs'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsBatchPredictionJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels a BatchPredictionJob. Starts asynchronous cancellation on the BatchPredictionJob. The server makes the best effort to cancel the job, but success is not guaranteed. Clients can use JobService.GetBatchPredictionJob or other methods to check whether the cancellation succeeded or whether the job completed despite cancellation. On a successful cancellation, the BatchPredictionJob is not deleted;instead its BatchPredictionJob.state is set to `CANCELLED`. Any files already outputted by the job are not deleted.

      Args:
        request: (AiplatformProjectsLocationsBatchPredictionJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/batchPredictionJobs/{batchPredictionJobsId}:cancel', http_method='POST', method_id='aiplatform.projects.locations.batchPredictionJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='googleCloudAiplatformV1CancelBatchPredictionJobRequest', request_type_name='AiplatformProjectsLocationsBatchPredictionJobsCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a BatchPredictionJob. A BatchPredictionJob once created will right away be attempted to start.

      Args:
        request: (AiplatformProjectsLocationsBatchPredictionJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1BatchPredictionJob) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/batchPredictionJobs', http_method='POST', method_id='aiplatform.projects.locations.batchPredictionJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/batchPredictionJobs', request_field='googleCloudAiplatformV1BatchPredictionJob', request_type_name='AiplatformProjectsLocationsBatchPredictionJobsCreateRequest', response_type_name='GoogleCloudAiplatformV1BatchPredictionJob', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a BatchPredictionJob. Can only be called on jobs that already finished.

      Args:
        request: (AiplatformProjectsLocationsBatchPredictionJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/batchPredictionJobs/{batchPredictionJobsId}', http_method='DELETE', method_id='aiplatform.projects.locations.batchPredictionJobs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsBatchPredictionJobsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a BatchPredictionJob.

      Args:
        request: (AiplatformProjectsLocationsBatchPredictionJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1BatchPredictionJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/batchPredictionJobs/{batchPredictionJobsId}', http_method='GET', method_id='aiplatform.projects.locations.batchPredictionJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsBatchPredictionJobsGetRequest', response_type_name='GoogleCloudAiplatformV1BatchPredictionJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists BatchPredictionJobs in a Location.

      Args:
        request: (AiplatformProjectsLocationsBatchPredictionJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListBatchPredictionJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/batchPredictionJobs', http_method='GET', method_id='aiplatform.projects.locations.batchPredictionJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/batchPredictionJobs', request_field='', request_type_name='AiplatformProjectsLocationsBatchPredictionJobsListRequest', response_type_name='GoogleCloudAiplatformV1ListBatchPredictionJobsResponse', supports_download=False)