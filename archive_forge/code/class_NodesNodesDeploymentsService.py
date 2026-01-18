from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class NodesNodesDeploymentsService(base_api.BaseApiService):
    """Service class for the nodes_nodes_deployments resource."""
    _NAME = 'nodes_nodes_deployments'

    def __init__(self, client):
        super(SasportalV1alpha1.NodesNodesDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new deployment.

      Args:
        request: (SasportalNodesNodesDeploymentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDeployment) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/nodes/{nodesId1}/deployments', http_method='POST', method_id='sasportal.nodes.nodes.deployments.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/deployments', request_field='sasPortalDeployment', request_type_name='SasportalNodesNodesDeploymentsCreateRequest', response_type_name='SasPortalDeployment', supports_download=False)

    def List(self, request, global_params=None):
        """Lists deployments.

      Args:
        request: (SasportalNodesNodesDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalListDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/nodes/{nodesId1}/deployments', http_method='GET', method_id='sasportal.nodes.nodes.deployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/deployments', request_field='', request_type_name='SasportalNodesNodesDeploymentsListRequest', response_type_name='SasPortalListDeploymentsResponse', supports_download=False)