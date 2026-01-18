from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SwitchToCustomMode(self, request, global_params=None):
    """Switches the network mode from auto subnet mode to custom subnet mode.

      Args:
        request: (ComputeNetworksSwitchToCustomModeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SwitchToCustomMode')
    return self._RunMethod(config, request, global_params=global_params)