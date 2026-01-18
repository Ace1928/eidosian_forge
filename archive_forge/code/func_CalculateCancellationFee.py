from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def CalculateCancellationFee(self, request, global_params=None):
    """Calculate cancellation fee for the specified commitment.

      Args:
        request: (ComputeRegionCommitmentsCalculateCancellationFeeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('CalculateCancellationFee')
    return self._RunMethod(config, request, global_params=global_params)