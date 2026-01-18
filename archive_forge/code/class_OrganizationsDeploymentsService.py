from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsDeploymentsService(base_api.BaseApiService):
    """Service class for the organizations_deployments resource."""
    _NAME = 'organizations_deployments'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all deployments of API proxies or shared flows.

      Args:
        request: (ApigeeOrganizationsDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/deployments', http_method='GET', method_id='apigee.organizations.deployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['sharedFlows'], relative_path='v1/{+parent}/deployments', request_field='', request_type_name='ApigeeOrganizationsDeploymentsListRequest', response_type_name='GoogleCloudApigeeV1ListDeploymentsResponse', supports_download=False)