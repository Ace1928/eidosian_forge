from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.websecurityscanner.v1beta import websecurityscanner_v1beta_messages as messages
class ProjectsScanConfigsScanRunsFindingsService(base_api.BaseApiService):
    """Service class for the projects_scanConfigs_scanRuns_findings resource."""
    _NAME = 'projects_scanConfigs_scanRuns_findings'

    def __init__(self, client):
        super(WebsecurityscannerV1beta.ProjectsScanConfigsScanRunsFindingsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a Finding.

      Args:
        request: (WebsecurityscannerProjectsScanConfigsScanRunsFindingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Finding) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/scanConfigs/{scanConfigsId}/scanRuns/{scanRunsId}/findings/{findingsId}', http_method='GET', method_id='websecurityscanner.projects.scanConfigs.scanRuns.findings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='WebsecurityscannerProjectsScanConfigsScanRunsFindingsGetRequest', response_type_name='Finding', supports_download=False)

    def List(self, request, global_params=None):
        """List Findings under a given ScanRun.

      Args:
        request: (WebsecurityscannerProjectsScanConfigsScanRunsFindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/scanConfigs/{scanConfigsId}/scanRuns/{scanRunsId}/findings', http_method='GET', method_id='websecurityscanner.projects.scanConfigs.scanRuns.findings.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/findings', request_field='', request_type_name='WebsecurityscannerProjectsScanConfigsScanRunsFindingsListRequest', response_type_name='ListFindingsResponse', supports_download=False)