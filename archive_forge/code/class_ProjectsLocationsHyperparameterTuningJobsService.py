from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsHyperparameterTuningJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_hyperparameterTuningJobs resource."""
    _NAME = 'projects_locations_hyperparameterTuningJobs'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsHyperparameterTuningJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels a HyperparameterTuningJob. Starts asynchronous cancellation on the HyperparameterTuningJob. The server makes a best effort to cancel the job, but success is not guaranteed. Clients can use JobService.GetHyperparameterTuningJob or other methods to check whether the cancellation succeeded or whether the job completed despite cancellation. On successful cancellation, the HyperparameterTuningJob is not deleted; instead it becomes a job with a HyperparameterTuningJob.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`, and HyperparameterTuningJob.state is set to `CANCELLED`.

      Args:
        request: (AiplatformProjectsLocationsHyperparameterTuningJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/hyperparameterTuningJobs/{hyperparameterTuningJobsId}:cancel', http_method='POST', method_id='aiplatform.projects.locations.hyperparameterTuningJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='googleCloudAiplatformV1CancelHyperparameterTuningJobRequest', request_type_name='AiplatformProjectsLocationsHyperparameterTuningJobsCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a HyperparameterTuningJob.

      Args:
        request: (AiplatformProjectsLocationsHyperparameterTuningJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1HyperparameterTuningJob) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/hyperparameterTuningJobs', http_method='POST', method_id='aiplatform.projects.locations.hyperparameterTuningJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/hyperparameterTuningJobs', request_field='googleCloudAiplatformV1HyperparameterTuningJob', request_type_name='AiplatformProjectsLocationsHyperparameterTuningJobsCreateRequest', response_type_name='GoogleCloudAiplatformV1HyperparameterTuningJob', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a HyperparameterTuningJob.

      Args:
        request: (AiplatformProjectsLocationsHyperparameterTuningJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/hyperparameterTuningJobs/{hyperparameterTuningJobsId}', http_method='DELETE', method_id='aiplatform.projects.locations.hyperparameterTuningJobs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsHyperparameterTuningJobsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a HyperparameterTuningJob.

      Args:
        request: (AiplatformProjectsLocationsHyperparameterTuningJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1HyperparameterTuningJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/hyperparameterTuningJobs/{hyperparameterTuningJobsId}', http_method='GET', method_id='aiplatform.projects.locations.hyperparameterTuningJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsHyperparameterTuningJobsGetRequest', response_type_name='GoogleCloudAiplatformV1HyperparameterTuningJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists HyperparameterTuningJobs in a Location.

      Args:
        request: (AiplatformProjectsLocationsHyperparameterTuningJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListHyperparameterTuningJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/hyperparameterTuningJobs', http_method='GET', method_id='aiplatform.projects.locations.hyperparameterTuningJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/hyperparameterTuningJobs', request_field='', request_type_name='AiplatformProjectsLocationsHyperparameterTuningJobsListRequest', response_type_name='GoogleCloudAiplatformV1ListHyperparameterTuningJobsResponse', supports_download=False)