from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class OrganizationsLocationsAuditReportsService(base_api.BaseApiService):
    """Service class for the organizations_locations_auditReports resource."""
    _NAME = 'organizations_locations_auditReports'

    def __init__(self, client):
        super(AuditmanagerV1alpha.OrganizationsLocationsAuditReportsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists audit reports in the selected parent scope.

      Args:
        request: (AuditmanagerOrganizationsLocationsAuditReportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAuditReportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/auditReports', http_method='GET', method_id='auditmanager.organizations.locations.auditReports.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/auditReports', request_field='', request_type_name='AuditmanagerOrganizationsLocationsAuditReportsListRequest', response_type_name='ListAuditReportsResponse', supports_download=False)