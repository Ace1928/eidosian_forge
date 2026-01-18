from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class CustomersNodesDeploymentsService(base_api.BaseApiService):
    """Service class for the customers_nodes_deployments resource."""
    _NAME = 'customers_nodes_deployments'

    def __init__(self, client):
        super(SasportalV1alpha1.CustomersNodesDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new deployment.

      Args:
        request: (SasportalCustomersNodesDeploymentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDeployment) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/nodes/{nodesId}/deployments', http_method='POST', method_id='sasportal.customers.nodes.deployments.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/deployments', request_field='sasPortalDeployment', request_type_name='SasportalCustomersNodesDeploymentsCreateRequest', response_type_name='SasPortalDeployment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists deployments.

      Args:
        request: (SasportalCustomersNodesDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalListDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/nodes/{nodesId}/deployments', http_method='GET', method_id='sasportal.customers.nodes.deployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/deployments', request_field='', request_type_name='SasportalCustomersNodesDeploymentsListRequest', response_type_name='SasPortalListDeploymentsResponse', supports_download=False)