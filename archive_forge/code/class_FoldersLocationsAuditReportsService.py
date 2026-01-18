from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class FoldersLocationsAuditReportsService(base_api.BaseApiService):
    """Service class for the folders_locations_auditReports resource."""
    _NAME = 'folders_locations_auditReports'

    def __init__(self, client):
        super(AuditmanagerV1alpha.FoldersLocationsAuditReportsService, self).__init__(client)
        self._upload_configs = {}

    def Generate(self, request, global_params=None):
        """Register the Audit Report generation requests and returns the OperationId using which the customer can track the report generation progress.

      Args:
        request: (AuditmanagerFoldersLocationsAuditReportsGenerateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Generate')
        return self._RunMethod(config, request, global_params=global_params)
    Generate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}/auditReports:generate', http_method='POST', method_id='auditmanager.folders.locations.auditReports.generate', ordered_params=['scope'], path_params=['scope'], query_params=[], relative_path='v1alpha/{+scope}/auditReports:generate', request_field='generateAuditReportRequest', request_type_name='AuditmanagerFoldersLocationsAuditReportsGenerateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Get the overall audit report.

      Args:
        request: (AuditmanagerFoldersLocationsAuditReportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuditReport) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}/auditReports/{auditReportsId}', http_method='GET', method_id='auditmanager.folders.locations.auditReports.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AuditmanagerFoldersLocationsAuditReportsGetRequest', response_type_name='AuditReport', supports_download=False)

    def List(self, request, global_params=None):
        """Lists audit reports in the selected parent scope.

      Args:
        request: (AuditmanagerFoldersLocationsAuditReportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAuditReportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}/auditReports', http_method='GET', method_id='auditmanager.folders.locations.auditReports.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/auditReports', request_field='', request_type_name='AuditmanagerFoldersLocationsAuditReportsListRequest', response_type_name='ListAuditReportsResponse', supports_download=False)