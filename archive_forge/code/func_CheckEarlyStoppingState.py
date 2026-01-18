from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.ml.v1 import ml_v1_messages as messages
def CheckEarlyStoppingState(self, request, global_params=None):
    """Checks whether a trial should stop or not. Returns a long-running operation. When the operation is successful, it will contain a CheckTrialEarlyStoppingStateResponse.

      Args:
        request: (MlProjectsLocationsStudiesTrialsCheckEarlyStoppingStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('CheckEarlyStoppingState')
    return self._RunMethod(config, request, global_params=global_params)