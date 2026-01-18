from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1beta import osconfig_v1beta_messages as messages
class ProjectsPatchJobsInstanceDetailsService(base_api.BaseApiService):
    """Service class for the projects_patchJobs_instanceDetails resource."""
    _NAME = 'projects_patchJobs_instanceDetails'

    def __init__(self, client):
        super(OsconfigV1beta.ProjectsPatchJobsInstanceDetailsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Get a list of instance details for a given patch job.

      Args:
        request: (OsconfigProjectsPatchJobsInstanceDetailsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPatchJobInstanceDetailsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchJobs/{patchJobsId}/instanceDetails', http_method='GET', method_id='osconfig.projects.patchJobs.instanceDetails.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/instanceDetails', request_field='', request_type_name='OsconfigProjectsPatchJobsInstanceDetailsListRequest', response_type_name='ListPatchJobInstanceDetailsResponse', supports_download=False)