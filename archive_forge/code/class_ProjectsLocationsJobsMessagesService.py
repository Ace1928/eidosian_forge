from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
class ProjectsLocationsJobsMessagesService(base_api.BaseApiService):
    """Service class for the projects_locations_jobs_messages resource."""
    _NAME = 'projects_locations_jobs_messages'

    def __init__(self, client):
        super(DataflowV1b3.ProjectsLocationsJobsMessagesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Request the job status. To request the status of a job, we recommend using `projects.locations.jobs.messages.list` with a [regional endpoint] (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints). Using `projects.jobs.messages.list` is not recommended, as you can only request the status of jobs that are running in `us-central1`.

      Args:
        request: (DataflowProjectsLocationsJobsMessagesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListJobMessagesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataflow.projects.locations.jobs.messages.list', ordered_params=['projectId', 'location', 'jobId'], path_params=['jobId', 'location', 'projectId'], query_params=['endTime', 'minimumImportance', 'pageSize', 'pageToken', 'startTime'], relative_path='v1b3/projects/{projectId}/locations/{location}/jobs/{jobId}/messages', request_field='', request_type_name='DataflowProjectsLocationsJobsMessagesListRequest', response_type_name='ListJobMessagesResponse', supports_download=False)