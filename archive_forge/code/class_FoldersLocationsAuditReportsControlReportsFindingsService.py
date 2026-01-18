from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class FoldersLocationsAuditReportsControlReportsFindingsService(base_api.BaseApiService):
    """Service class for the folders_locations_auditReports_controlReports_findings resource."""
    _NAME = 'folders_locations_auditReports_controlReports_findings'

    def __init__(self, client):
        super(AuditmanagerV1alpha.FoldersLocationsAuditReportsControlReportsFindingsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get the finding from a control report.

      Args:
        request: (AuditmanagerFoldersLocationsAuditReportsControlReportsFindingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Finding) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}/auditReports/{auditReportsId}/controlReports/{controlReportsId}/findings/{findingsId}', http_method='GET', method_id='auditmanager.folders.locations.auditReports.controlReports.findings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AuditmanagerFoldersLocationsAuditReportsControlReportsFindingsGetRequest', response_type_name='Finding', supports_download=False)

    def List(self, request, global_params=None):
        """Fetches all findings under the control report.

      Args:
        request: (AuditmanagerFoldersLocationsAuditReportsControlReportsFindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}/auditReports/{auditReportsId}/controlReports/{controlReportsId}/findings', http_method='GET', method_id='auditmanager.folders.locations.auditReports.controlReports.findings.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/findings', request_field='', request_type_name='AuditmanagerFoldersLocationsAuditReportsControlReportsFindingsListRequest', response_type_name='ListFindingsResponse', supports_download=False)