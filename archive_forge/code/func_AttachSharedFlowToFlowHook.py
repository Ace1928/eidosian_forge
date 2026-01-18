from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def AttachSharedFlowToFlowHook(self, request, global_params=None):
    """Attaches a shared flow to a flow hook.

      Args:
        request: (ApigeeOrganizationsEnvironmentsFlowhooksAttachSharedFlowToFlowHookRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1FlowHook) The response message.
      """
    config = self.GetMethodConfig('AttachSharedFlowToFlowHook')
    return self._RunMethod(config, request, global_params=global_params)