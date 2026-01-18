from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.auditmanager.v1alpha import auditmanager_v1alpha_messages as messages
class OrganizationsLocationsResourceEnrollmentStatusesService(base_api.BaseApiService):
    """Service class for the organizations_locations_resourceEnrollmentStatuses resource."""
    _NAME = 'organizations_locations_resourceEnrollmentStatuses'

    def __init__(self, client):
        super(AuditmanagerV1alpha.OrganizationsLocationsResourceEnrollmentStatusesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Fetches all resources under the parent along with their enrollment.

      Args:
        request: (AuditmanagerOrganizationsLocationsResourceEnrollmentStatusesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListResourceEnrollmentStatusesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/resourceEnrollmentStatuses', http_method='GET', method_id='auditmanager.organizations.locations.resourceEnrollmentStatuses.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/resourceEnrollmentStatuses', request_field='', request_type_name='AuditmanagerOrganizationsLocationsResourceEnrollmentStatusesListRequest', response_type_name='ListResourceEnrollmentStatusesResponse', supports_download=False)