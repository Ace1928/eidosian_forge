from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class CustomersNodesNodesService(base_api.BaseApiService):
    """Service class for the customers_nodes_nodes resource."""
    _NAME = 'customers_nodes_nodes'

    def __init__(self, client):
        super(SasportalV1alpha1.CustomersNodesNodesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new node.

      Args:
        request: (SasportalCustomersNodesNodesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalNode) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/nodes/{nodesId}/nodes', http_method='POST', method_id='sasportal.customers.nodes.nodes.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/nodes', request_field='sasPortalNode', request_type_name='SasportalCustomersNodesNodesCreateRequest', response_type_name='SasPortalNode', supports_download=False)

    def List(self, request, global_params=None):
        """Lists nodes.

      Args:
        request: (SasportalCustomersNodesNodesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalListNodesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/customers/{customersId}/nodes/{nodesId}/nodes', http_method='GET', method_id='sasportal.customers.nodes.nodes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/nodes', request_field='', request_type_name='SasportalCustomersNodesNodesListRequest', response_type_name='SasPortalListNodesResponse', supports_download=False)