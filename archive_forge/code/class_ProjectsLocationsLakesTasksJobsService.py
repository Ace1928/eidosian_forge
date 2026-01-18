from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
class ProjectsLocationsLakesTasksJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_lakes_tasks_jobs resource."""
    _NAME = 'projects_locations_lakes_tasks_jobs'

    def __init__(self, client):
        super(DataplexV1.ProjectsLocationsLakesTasksJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancel jobs running for the task resource.

      Args:
        request: (DataplexProjectsLocationsLakesTasksJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks/{tasksId}/jobs/{jobsId}:cancel', http_method='POST', method_id='dataplex.projects.locations.lakes.tasks.jobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='googleCloudDataplexV1CancelJobRequest', request_type_name='DataplexProjectsLocationsLakesTasksJobsCancelRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Get job resource.

      Args:
        request: (DataplexProjectsLocationsLakesTasksJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1Job) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks/{tasksId}/jobs/{jobsId}', http_method='GET', method_id='dataplex.projects.locations.lakes.tasks.jobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataplexProjectsLocationsLakesTasksJobsGetRequest', response_type_name='GoogleCloudDataplexV1Job', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Jobs under the given task.

      Args:
        request: (DataplexProjectsLocationsLakesTasksJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1ListJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/lakes/{lakesId}/tasks/{tasksId}/jobs', http_method='GET', method_id='dataplex.projects.locations.lakes.tasks.jobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/jobs', request_field='', request_type_name='DataplexProjectsLocationsLakesTasksJobsListRequest', response_type_name='GoogleCloudDataplexV1ListJobsResponse', supports_download=False)