from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
def DeleteContexts(self, request, global_params=None):
    """Deletes all active contexts in the specified session.

      Args:
        request: (DialogflowProjectsLocationsAgentSessionsDeleteContextsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
    config = self.GetMethodConfig('DeleteContexts')
    return self._RunMethod(config, request, global_params=global_params)