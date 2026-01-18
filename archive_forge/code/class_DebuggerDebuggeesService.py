from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddebugger.v2 import clouddebugger_v2_messages as messages
class DebuggerDebuggeesService(base_api.BaseApiService):
    """Service class for the debugger_debuggees resource."""
    _NAME = 'debugger_debuggees'

    def __init__(self, client):
        super(ClouddebuggerV2.DebuggerDebuggeesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all the debuggees that the user has access to.

      Args:
        request: (ClouddebuggerDebuggerDebuggeesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDebuggeesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='clouddebugger.debugger.debuggees.list', ordered_params=[], path_params=[], query_params=['clientVersion', 'includeInactive', 'project'], relative_path='v2/debugger/debuggees', request_field='', request_type_name='ClouddebuggerDebuggerDebuggeesListRequest', response_type_name='ListDebuggeesResponse', supports_download=False)