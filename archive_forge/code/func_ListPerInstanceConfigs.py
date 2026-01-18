from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListPerInstanceConfigs(self, request, global_params=None):
    """Lists all of the per-instance configurations defined for the managed instance group. The orderBy query parameter is not supported.

      Args:
        request: (ComputeRegionInstanceGroupManagersListPerInstanceConfigsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionInstanceGroupManagersListInstanceConfigsResp) The response message.
      """
    config = self.GetMethodConfig('ListPerInstanceConfigs')
    return self._RunMethod(config, request, global_params=global_params)