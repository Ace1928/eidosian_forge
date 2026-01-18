from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsCustomJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_customJobs resource."""
    _NAME = 'projects_locations_customJobs'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsCustomJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels a CustomJob. Starts asynchronous cancellation on the CustomJob. The server makes a best effort to cancel the job, but success is not guaranteed. Clients can use JobService.GetCustomJob or other methods to check whether the cancellation succeeded or whether the job completed despite cancellation. On successful cancellation, the CustomJob is not deleted; instead it becomes a job with a CustomJob.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`, and CustomJob.state is set to `CANCELLED`.

      Args:
        request: (AiplatformProjectsLocationsCustomJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customJobs/{customJobsId}:cancel', http_method='POST', method_id='aiplatform.projects.locations.customJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='googleCloudAiplatformV1CancelCustomJobRequest', request_type_name='AiplatformProjectsLocationsCustomJobsCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a CustomJob. A created CustomJob right away will be attempted to be run.

      Args:
        request: (AiplatformProjectsLocationsCustomJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1CustomJob) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customJobs', http_method='POST', method_id='aiplatform.projects.locations.customJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/customJobs', request_field='googleCloudAiplatformV1CustomJob', request_type_name='AiplatformProjectsLocationsCustomJobsCreateRequest', response_type_name='GoogleCloudAiplatformV1CustomJob', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a CustomJob.

      Args:
        request: (AiplatformProjectsLocationsCustomJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customJobs/{customJobsId}', http_method='DELETE', method_id='aiplatform.projects.locations.customJobs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsCustomJobsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a CustomJob.

      Args:
        request: (AiplatformProjectsLocationsCustomJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1CustomJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customJobs/{customJobsId}', http_method='GET', method_id='aiplatform.projects.locations.customJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsCustomJobsGetRequest', response_type_name='GoogleCloudAiplatformV1CustomJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CustomJobs in a Location.

      Args:
        request: (AiplatformProjectsLocationsCustomJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListCustomJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/customJobs', http_method='GET', method_id='aiplatform.projects.locations.customJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/customJobs', request_field='', request_type_name='AiplatformProjectsLocationsCustomJobsListRequest', response_type_name='GoogleCloudAiplatformV1ListCustomJobsResponse', supports_download=False)