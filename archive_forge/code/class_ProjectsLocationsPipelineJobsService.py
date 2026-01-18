from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsPipelineJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_pipelineJobs resource."""
    _NAME = 'projects_locations_pipelineJobs'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsPipelineJobsService, self).__init__(client)
        self._upload_configs = {}

    def BatchCancel(self, request, global_params=None):
        """Batch cancel PipelineJobs. Firstly the server will check if all the jobs are in non-terminal states, and skip the jobs that are already terminated. If the operation failed, none of the pipeline jobs are cancelled. The server will poll the states of all the pipeline jobs periodically to check the cancellation status. This operation will return an LRO.

      Args:
        request: (AiplatformProjectsLocationsPipelineJobsBatchCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchCancel')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelineJobs:batchCancel', http_method='POST', method_id='aiplatform.projects.locations.pipelineJobs.batchCancel', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/pipelineJobs:batchCancel', request_field='googleCloudAiplatformV1BatchCancelPipelineJobsRequest', request_type_name='AiplatformProjectsLocationsPipelineJobsBatchCancelRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def BatchDelete(self, request, global_params=None):
        """Batch deletes PipelineJobs The Operation is atomic. If it fails, none of the PipelineJobs are deleted. If it succeeds, all of the PipelineJobs are deleted.

      Args:
        request: (AiplatformProjectsLocationsPipelineJobsBatchDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchDelete')
        return self._RunMethod(config, request, global_params=global_params)
    BatchDelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelineJobs:batchDelete', http_method='POST', method_id='aiplatform.projects.locations.pipelineJobs.batchDelete', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/pipelineJobs:batchDelete', request_field='googleCloudAiplatformV1BatchDeletePipelineJobsRequest', request_type_name='AiplatformProjectsLocationsPipelineJobsBatchDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Cancel(self, request, global_params=None):
        """Cancels a PipelineJob. Starts asynchronous cancellation on the PipelineJob. The server makes a best effort to cancel the pipeline, but success is not guaranteed. Clients can use PipelineService.GetPipelineJob or other methods to check whether the cancellation succeeded or whether the pipeline completed despite cancellation. On successful cancellation, the PipelineJob is not deleted; instead it becomes a pipeline with a PipelineJob.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`, and PipelineJob.state is set to `CANCELLED`.

      Args:
        request: (AiplatformProjectsLocationsPipelineJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelineJobs/{pipelineJobsId}:cancel', http_method='POST', method_id='aiplatform.projects.locations.pipelineJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='googleCloudAiplatformV1CancelPipelineJobRequest', request_type_name='AiplatformProjectsLocationsPipelineJobsCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a PipelineJob. A PipelineJob will run immediately when created.

      Args:
        request: (AiplatformProjectsLocationsPipelineJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1PipelineJob) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelineJobs', http_method='POST', method_id='aiplatform.projects.locations.pipelineJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=['pipelineJobId'], relative_path='v1/{+parent}/pipelineJobs', request_field='googleCloudAiplatformV1PipelineJob', request_type_name='AiplatformProjectsLocationsPipelineJobsCreateRequest', response_type_name='GoogleCloudAiplatformV1PipelineJob', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a PipelineJob.

      Args:
        request: (AiplatformProjectsLocationsPipelineJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelineJobs/{pipelineJobsId}', http_method='DELETE', method_id='aiplatform.projects.locations.pipelineJobs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsPipelineJobsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a PipelineJob.

      Args:
        request: (AiplatformProjectsLocationsPipelineJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1PipelineJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelineJobs/{pipelineJobsId}', http_method='GET', method_id='aiplatform.projects.locations.pipelineJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsPipelineJobsGetRequest', response_type_name='GoogleCloudAiplatformV1PipelineJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists PipelineJobs in a Location.

      Args:
        request: (AiplatformProjectsLocationsPipelineJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListPipelineJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/pipelineJobs', http_method='GET', method_id='aiplatform.projects.locations.pipelineJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/pipelineJobs', request_field='', request_type_name='AiplatformProjectsLocationsPipelineJobsListRequest', response_type_name='GoogleCloudAiplatformV1ListPipelineJobsResponse', supports_download=False)