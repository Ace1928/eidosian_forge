from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def PatchPerInstanceConfigs(self, request, global_params=None):
    """Inserts or patches per-instance configurations for the managed instance group. perInstanceConfig.name serves as a key used to distinguish whether to perform insert or patch.

      Args:
        request: (ComputeRegionInstanceGroupManagersPatchPerInstanceConfigsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('PatchPerInstanceConfigs')
    return self._RunMethod(config, request, global_params=global_params)