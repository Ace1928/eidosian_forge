from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class OrganizationsLocationsStandardsControlsService(base_api.BaseApiService):
    """Service class for the organizations_locations_standards_controls resource."""
    _NAME = 'organizations_locations_standards_controls'

    def __init__(self, client):
        super(AuditmanagerV1alpha.OrganizationsLocationsStandardsControlsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Gets controls needed to be implemented to be compliant to a standard.

      Args:
        request: (AuditmanagerOrganizationsLocationsStandardsControlsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListControlsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/standards/{standardsId}/controls', http_method='GET', method_id='auditmanager.organizations.locations.standards.controls.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/controls', request_field='', request_type_name='AuditmanagerOrganizationsLocationsStandardsControlsListRequest', response_type_name='ListControlsResponse', supports_download=False)