from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class NodeTypesService(base_api.BaseApiService):
    """Service class for the nodeTypes resource."""
    _NAME = 'nodeTypes'

    def __init__(self, client):
        super(ComputeBeta.NodeTypesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of node types. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeNodeTypesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeTypeAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.nodeTypes.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/nodeTypes', request_field='', request_type_name='ComputeNodeTypesAggregatedListRequest', response_type_name='NodeTypeAggregatedList', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified node type.

      Args:
        request: (ComputeNodeTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.nodeTypes.get', ordered_params=['project', 'zone', 'nodeType'], path_params=['nodeType', 'project', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/nodeTypes/{nodeType}', request_field='', request_type_name='ComputeNodeTypesGetRequest', response_type_name='NodeType', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of node types available to the specified project.

      Args:
        request: (ComputeNodeTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NodeTypeList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.nodeTypes.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/nodeTypes', request_field='', request_type_name='ComputeNodeTypesListRequest', response_type_name='NodeTypeList', supports_download=False)