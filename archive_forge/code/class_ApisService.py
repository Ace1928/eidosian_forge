from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.discovery.v1 import discovery_v1_messages as messages
class ApisService(base_api.BaseApiService):
    """Service class for the apis resource."""
    _NAME = 'apis'

    def __init__(self, client):
        super(DiscoveryV1.ApisService, self).__init__(client)
        self._upload_configs = {}

    def GetRest(self, request, global_params=None):
        """Retrieve the description of a particular version of an api.

      Args:
        request: (DiscoveryApisGetRestRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RestDescription) The response message.
      """
        config = self.GetMethodConfig('GetRest')
        return self._RunMethod(config, request, global_params=global_params)
    GetRest.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='discovery.apis.getRest', ordered_params=['api', 'version'], path_params=['api', 'version'], query_params=[], relative_path='apis/{api}/{version}/rest', request_field='', request_type_name='DiscoveryApisGetRestRequest', response_type_name='RestDescription', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieve the list of APIs supported at this endpoint.

      Args:
        request: (DiscoveryApisListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DirectoryList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='discovery.apis.list', ordered_params=[], path_params=[], query_params=['label', 'name', 'preferred'], relative_path='apis', request_field='', request_type_name='DiscoveryApisListRequest', response_type_name='DirectoryList', supports_download=False)