from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SendDiagnosticInterrupt(self, request, global_params=None):
    """Sends diagnostic interrupt to the instance.

      Args:
        request: (ComputeInstancesSendDiagnosticInterruptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeInstancesSendDiagnosticInterruptResponse) The response message.
      """
    config = self.GetMethodConfig('SendDiagnosticInterrupt')
    return self._RunMethod(config, request, global_params=global_params)