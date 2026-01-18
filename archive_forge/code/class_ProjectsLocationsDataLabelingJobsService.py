from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsDataLabelingJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_dataLabelingJobs resource."""
    _NAME = 'projects_locations_dataLabelingJobs'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsDataLabelingJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels a DataLabelingJob. Success of cancellation is not guaranteed.

      Args:
        request: (AiplatformProjectsLocationsDataLabelingJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataLabelingJobs/{dataLabelingJobsId}:cancel', http_method='POST', method_id='aiplatform.projects.locations.dataLabelingJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='googleCloudAiplatformV1CancelDataLabelingJobRequest', request_type_name='AiplatformProjectsLocationsDataLabelingJobsCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a DataLabelingJob.

      Args:
        request: (AiplatformProjectsLocationsDataLabelingJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1DataLabelingJob) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataLabelingJobs', http_method='POST', method_id='aiplatform.projects.locations.dataLabelingJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/dataLabelingJobs', request_field='googleCloudAiplatformV1DataLabelingJob', request_type_name='AiplatformProjectsLocationsDataLabelingJobsCreateRequest', response_type_name='GoogleCloudAiplatformV1DataLabelingJob', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a DataLabelingJob.

      Args:
        request: (AiplatformProjectsLocationsDataLabelingJobsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataLabelingJobs/{dataLabelingJobsId}', http_method='DELETE', method_id='aiplatform.projects.locations.dataLabelingJobs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDataLabelingJobsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a DataLabelingJob.

      Args:
        request: (AiplatformProjectsLocationsDataLabelingJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1DataLabelingJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataLabelingJobs/{dataLabelingJobsId}', http_method='GET', method_id='aiplatform.projects.locations.dataLabelingJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDataLabelingJobsGetRequest', response_type_name='GoogleCloudAiplatformV1DataLabelingJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DataLabelingJobs in a Location.

      Args:
        request: (AiplatformProjectsLocationsDataLabelingJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListDataLabelingJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataLabelingJobs', http_method='GET', method_id='aiplatform.projects.locations.dataLabelingJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/dataLabelingJobs', request_field='', request_type_name='AiplatformProjectsLocationsDataLabelingJobsListRequest', response_type_name='GoogleCloudAiplatformV1ListDataLabelingJobsResponse', supports_download=False)