from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def DetachSharedFlowFromFlowHook(self, request, global_params=None):
    """Detaches a shared flow from a flow hook.

      Args:
        request: (ApigeeOrganizationsEnvironmentsFlowhooksDetachSharedFlowFromFlowHookRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1FlowHook) The response message.
      """
    config = self.GetMethodConfig('DetachSharedFlowFromFlowHook')
    return self._RunMethod(config, request, global_params=global_params)