from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class FoldersLocationsAuditScopeReportsService(base_api.BaseApiService):
    """Service class for the folders_locations_auditScopeReports resource."""
    _NAME = 'folders_locations_auditScopeReports'

    def __init__(self, client):
        super(AuditmanagerV1alpha.FoldersLocationsAuditScopeReportsService, self).__init__(client)
        self._upload_configs = {}

    def Generate(self, request, global_params=None):
        """Generates a demo report highlighting different responsibilities (Google/Customer/ shared) required to be fulfilled for the customer's workload to be compliant with the given standard.

      Args:
        request: (AuditmanagerFoldersLocationsAuditScopeReportsGenerateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuditScopeReport) The response message.
      """
        config = self.GetMethodConfig('Generate')
        return self._RunMethod(config, request, global_params=global_params)
    Generate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/folders/{foldersId}/locations/{locationsId}/auditScopeReports:generate', http_method='POST', method_id='auditmanager.folders.locations.auditScopeReports.generate', ordered_params=['scope'], path_params=['scope'], query_params=[], relative_path='v1alpha/{+scope}/auditScopeReports:generate', request_field='generateAuditScopeReportRequest', request_type_name='AuditmanagerFoldersLocationsAuditScopeReportsGenerateRequest', response_type_name='AuditScopeReport', supports_download=False)