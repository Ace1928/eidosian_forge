from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
class NetworkPlacementsService(base_api.BaseApiService):
    """Service class for the networkPlacements resource."""
    _NAME = 'networkPlacements'

    def __init__(self, client):
        super(ComputeAlpha.NetworkPlacementsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns the specified network placement.

      Args:
        request: (ComputeNetworkPlacementsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkPlacement) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkPlacements.get', ordered_params=['project', 'networkPlacement'], path_params=['networkPlacement', 'project'], query_params=[], relative_path='projects/{project}/global/networkPlacements/{networkPlacement}', request_field='', request_type_name='ComputeNetworkPlacementsGetRequest', response_type_name='NetworkPlacement', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of network placements available to the specified project.

      Args:
        request: (ComputeNetworkPlacementsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkPlacementsListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkPlacements.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/networkPlacements', request_field='', request_type_name='ComputeNetworkPlacementsListRequest', response_type_name='NetworkPlacementsListResponse', supports_download=False)