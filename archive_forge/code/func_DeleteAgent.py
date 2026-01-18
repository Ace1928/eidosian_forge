from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
def DeleteAgent(self, request, global_params=None):
    """Deletes the specified agent.

      Args:
        request: (DialogflowProjectsDeleteAgentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
    config = self.GetMethodConfig('DeleteAgent')
    return self._RunMethod(config, request, global_params=global_params)