from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class TriggersService(base_api.BaseApiService):
    """Service class for the triggers resource."""
    _NAME = 'triggers'

    def __init__(self, client):
        super(AnthoseventsV1.TriggersService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Rpc to list triggers in all namespaces.

      Args:
        request: (AnthoseventsTriggersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTriggersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='anthosevents.triggers.list', ordered_params=[], path_params=[], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'pageSize', 'parent', 'resourceVersion', 'watch'], relative_path='apis/eventing.knative.dev/v1/triggers', request_field='', request_type_name='AnthoseventsTriggersListRequest', response_type_name='ListTriggersResponse', supports_download=False)