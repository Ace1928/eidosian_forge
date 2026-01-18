from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def CheckTrialEarlyStoppingState(self, request, global_params=None):
    """Checks whether a Trial should stop or not. Returns a long-running operation. When the operation is successful, it will contain a CheckTrialEarlyStoppingStateResponse.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsCheckTrialEarlyStoppingStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('CheckTrialEarlyStoppingState')
    return self._RunMethod(config, request, global_params=global_params)