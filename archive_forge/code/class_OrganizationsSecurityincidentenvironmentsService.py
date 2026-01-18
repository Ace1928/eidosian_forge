from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsSecurityincidentenvironmentsService(base_api.BaseApiService):
    """Service class for the organizations_securityincidentenvironments resource."""
    _NAME = 'organizations_securityincidentenvironments'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsSecurityincidentenvironmentsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all the Environments in an organization with Security Incident Stats.

      Args:
        request: (ApigeeOrganizationsSecurityincidentenvironmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListSecurityIncidentEnvironmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityincidentenvironments', http_method='GET', method_id='apigee.organizations.securityincidentenvironments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/securityincidentenvironments', request_field='', request_type_name='ApigeeOrganizationsSecurityincidentenvironmentsListRequest', response_type_name='GoogleCloudApigeeV1ListSecurityIncidentEnvironmentsResponse', supports_download=False)