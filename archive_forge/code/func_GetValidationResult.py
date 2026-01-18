from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
def GetValidationResult(self, request, global_params=None):
    """Gets agent validation result. Agent validation is performed during training time and is updated automatically when training is completed.

      Args:
        request: (DialogflowProjectsLocationsAgentGetValidationResultRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ValidationResult) The response message.
      """
    config = self.GetMethodConfig('GetValidationResult')
    return self._RunMethod(config, request, global_params=global_params)