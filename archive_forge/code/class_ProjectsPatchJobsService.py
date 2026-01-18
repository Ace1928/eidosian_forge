from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1beta import osconfig_v1beta_messages as messages
class ProjectsPatchJobsService(base_api.BaseApiService):
    """Service class for the projects_patchJobs resource."""
    _NAME = 'projects_patchJobs'

    def __init__(self, client):
        super(OsconfigV1beta.ProjectsPatchJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancel a patch job. The patch job must be active. Canceled patch jobs cannot be restarted.

      Args:
        request: (OsconfigProjectsPatchJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PatchJob) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchJobs/{patchJobsId}:cancel', http_method='POST', method_id='osconfig.projects.patchJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:cancel', request_field='cancelPatchJobRequest', request_type_name='OsconfigProjectsPatchJobsCancelRequest', response_type_name='PatchJob', supports_download=False)

    def Execute(self, request, global_params=None):
        """Patch VM instances by creating and running a patch job.

      Args:
        request: (OsconfigProjectsPatchJobsExecuteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PatchJob) The response message.
      """
        config = self.GetMethodConfig('Execute')
        return self._RunMethod(config, request, global_params=global_params)
    Execute.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchJobs:execute', http_method='POST', method_id='osconfig.projects.patchJobs.execute', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta/{+parent}/patchJobs:execute', request_field='executePatchJobRequest', request_type_name='OsconfigProjectsPatchJobsExecuteRequest', response_type_name='PatchJob', supports_download=False)

    def Get(self, request, global_params=None):
        """Get the patch job. This can be used to track the progress of an ongoing patch job or review the details of completed jobs.

      Args:
        request: (OsconfigProjectsPatchJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PatchJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchJobs/{patchJobsId}', http_method='GET', method_id='osconfig.projects.patchJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='OsconfigProjectsPatchJobsGetRequest', response_type_name='PatchJob', supports_download=False)

    def List(self, request, global_params=None):
        """Get a list of patch jobs.

      Args:
        request: (OsconfigProjectsPatchJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPatchJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/patchJobs', http_method='GET', method_id='osconfig.projects.patchJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/patchJobs', request_field='', request_type_name='OsconfigProjectsPatchJobsListRequest', response_type_name='ListPatchJobsResponse', supports_download=False)