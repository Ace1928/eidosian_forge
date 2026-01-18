from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
class ProjectsLocationsJobsSnapshotsService(base_api.BaseApiService):
    """Service class for the projects_locations_jobs_snapshots resource."""
    _NAME = 'projects_locations_jobs_snapshots'

    def __init__(self, client):
        super(DataflowV1b3.ProjectsLocationsJobsSnapshotsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists snapshots.

      Args:
        request: (DataflowProjectsLocationsJobsSnapshotsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSnapshotsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataflow.projects.locations.jobs.snapshots.list', ordered_params=['projectId', 'location', 'jobId'], path_params=['jobId', 'location', 'projectId'], query_params=[], relative_path='v1b3/projects/{projectId}/locations/{location}/jobs/{jobId}/snapshots', request_field='', request_type_name='DataflowProjectsLocationsJobsSnapshotsListRequest', response_type_name='ListSnapshotsResponse', supports_download=False)