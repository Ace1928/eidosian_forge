from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class NodesService(base_api.BaseApiService):
    """Service class for the nodes resource."""
    _NAME = 'nodes'

    def __init__(self, client):
        super(SasportalV1alpha1.NodesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns a requested node.

      Args:
        request: (SasportalNodesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalNode) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}', http_method='GET', method_id='sasportal.nodes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='SasportalNodesGetRequest', response_type_name='SasPortalNode', supports_download=False)