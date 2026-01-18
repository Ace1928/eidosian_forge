from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetMacsecConfig(self, request, global_params=None):
    """Returns the interconnectMacsecConfig for the specified Interconnect.

      Args:
        request: (ComputeInterconnectsGetMacsecConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InterconnectsGetMacsecConfigResponse) The response message.
      """
    config = self.GetMethodConfig('GetMacsecConfig')
    return self._RunMethod(config, request, global_params=global_params)