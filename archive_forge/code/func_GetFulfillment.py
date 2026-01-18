from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
def GetFulfillment(self, request, global_params=None):
    """Retrieves the fulfillment.

      Args:
        request: (DialogflowProjectsLocationsAgentGetFulfillmentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Fulfillment) The response message.
      """
    config = self.GetMethodConfig('GetFulfillment')
    return self._RunMethod(config, request, global_params=global_params)