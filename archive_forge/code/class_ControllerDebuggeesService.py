from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddebugger.v2 import clouddebugger_v2_messages as messages
class ControllerDebuggeesService(base_api.BaseApiService):
    """Service class for the controller_debuggees resource."""
    _NAME = 'controller_debuggees'

    def __init__(self, client):
        super(ClouddebuggerV2.ControllerDebuggeesService, self).__init__(client)
        self._upload_configs = {}

    def Register(self, request, global_params=None):
        """Registers the debuggee with the controller service. All agents attached to the same application must call this method with exactly the same request content to get back the same stable `debuggee_id`. Agents should call this method again whenever `google.rpc.Code.NOT_FOUND` is returned from any controller method. This protocol allows the controller service to disable debuggees, recover from data loss, or change the `debuggee_id` format. Agents must handle `debuggee_id` value changing upon re-registration.

      Args:
        request: (RegisterDebuggeeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegisterDebuggeeResponse) The response message.
      """
        config = self.GetMethodConfig('Register')
        return self._RunMethod(config, request, global_params=global_params)
    Register.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='clouddebugger.controller.debuggees.register', ordered_params=[], path_params=[], query_params=[], relative_path='v2/controller/debuggees/register', request_field='<request>', request_type_name='RegisterDebuggeeRequest', response_type_name='RegisterDebuggeeResponse', supports_download=False)