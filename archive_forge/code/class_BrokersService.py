from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class BrokersService(base_api.BaseApiService):
    """Service class for the brokers resource."""
    _NAME = 'brokers'

    def __init__(self, client):
        super(AnthoseventsV1.BrokersService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Rpc to list brokers in all namespaces.

      Args:
        request: (AnthoseventsBrokersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBrokersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='anthosevents.brokers.list', ordered_params=[], path_params=[], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'parent', 'resourceVersion', 'watch'], relative_path='apis/eventing.knative.dev/v1/brokers', request_field='', request_type_name='AnthoseventsBrokersListRequest', response_type_name='ListBrokersResponse', supports_download=False)