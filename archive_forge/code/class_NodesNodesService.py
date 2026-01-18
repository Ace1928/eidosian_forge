from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class NodesNodesService(base_api.BaseApiService):
    """Service class for the nodes_nodes resource."""
    _NAME = 'nodes_nodes'

    def __init__(self, client):
        super(SasportalV1alpha1.NodesNodesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new node.

      Args:
        request: (SasportalNodesNodesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalNode) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/nodes', http_method='POST', method_id='sasportal.nodes.nodes.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/nodes', request_field='sasPortalNode', request_type_name='SasportalNodesNodesCreateRequest', response_type_name='SasPortalNode', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a node.

      Args:
        request: (SasportalNodesNodesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/nodes/{nodesId1}', http_method='DELETE', method_id='sasportal.nodes.nodes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SasportalNodesNodesDeleteRequest', response_type_name='SasPortalEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns a requested node.

      Args:
        request: (SasportalNodesNodesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalNode) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/nodes/{nodesId1}', http_method='GET', method_id='sasportal.nodes.nodes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SasportalNodesNodesGetRequest', response_type_name='SasPortalNode', supports_download=False)

    def List(self, request, global_params=None):
        """Lists nodes.

      Args:
        request: (SasportalNodesNodesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalListNodesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/nodes', http_method='GET', method_id='sasportal.nodes.nodes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/nodes', request_field='', request_type_name='SasportalNodesNodesListRequest', response_type_name='SasPortalListNodesResponse', supports_download=False)

    def Move(self, request, global_params=None):
        """Moves a node under another node or customer.

      Args:
        request: (SasportalNodesNodesMoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalOperation) The response message.
      """
        config = self.GetMethodConfig('Move')
        return self._RunMethod(config, request, global_params=global_params)
    Move.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/nodes/{nodesId1}:move', http_method='POST', method_id='sasportal.nodes.nodes.move', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:move', request_field='sasPortalMoveNodeRequest', request_type_name='SasportalNodesNodesMoveRequest', response_type_name='SasPortalOperation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing node.

      Args:
        request: (SasportalNodesNodesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalNode) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/nodes/{nodesId1}', http_method='PATCH', method_id='sasportal.nodes.nodes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='sasPortalNode', request_type_name='SasportalNodesNodesPatchRequest', response_type_name='SasPortalNode', supports_download=False)