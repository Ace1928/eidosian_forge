from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
def GetAgent(self, request, global_params=None):
    """Retrieves the specified agent.

      Args:
        request: (DialogflowProjectsGetAgentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Agent) The response message.
      """
    config = self.GetMethodConfig('GetAgent')
    return self._RunMethod(config, request, global_params=global_params)