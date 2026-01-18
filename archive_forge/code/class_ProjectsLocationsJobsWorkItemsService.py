from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
class ProjectsLocationsJobsWorkItemsService(base_api.BaseApiService):
    """Service class for the projects_locations_jobs_workItems resource."""
    _NAME = 'projects_locations_jobs_workItems'

    def __init__(self, client):
        super(DataflowV1b3.ProjectsLocationsJobsWorkItemsService, self).__init__(client)
        self._upload_configs = {}

    def Lease(self, request, global_params=None):
        """Leases a dataflow WorkItem to run.

      Args:
        request: (DataflowProjectsLocationsJobsWorkItemsLeaseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LeaseWorkItemResponse) The response message.
      """
        config = self.GetMethodConfig('Lease')
        return self._RunMethod(config, request, global_params=global_params)
    Lease.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataflow.projects.locations.jobs.workItems.lease', ordered_params=['projectId', 'location', 'jobId'], path_params=['jobId', 'location', 'projectId'], query_params=[], relative_path='v1b3/projects/{projectId}/locations/{location}/jobs/{jobId}/workItems:lease', request_field='leaseWorkItemRequest', request_type_name='DataflowProjectsLocationsJobsWorkItemsLeaseRequest', response_type_name='LeaseWorkItemResponse', supports_download=False)

    def ReportStatus(self, request, global_params=None):
        """Reports the status of dataflow WorkItems leased by a worker.

      Args:
        request: (DataflowProjectsLocationsJobsWorkItemsReportStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReportWorkItemStatusResponse) The response message.
      """
        config = self.GetMethodConfig('ReportStatus')
        return self._RunMethod(config, request, global_params=global_params)
    ReportStatus.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataflow.projects.locations.jobs.workItems.reportStatus', ordered_params=['projectId', 'location', 'jobId'], path_params=['jobId', 'location', 'projectId'], query_params=[], relative_path='v1b3/projects/{projectId}/locations/{location}/jobs/{jobId}/workItems:reportStatus', request_field='reportWorkItemStatusRequest', request_type_name='DataflowProjectsLocationsJobsWorkItemsReportStatusRequest', response_type_name='ReportWorkItemStatusResponse', supports_download=False)