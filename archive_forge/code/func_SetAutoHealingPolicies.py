from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetAutoHealingPolicies(self, request, global_params=None):
    """Modifies the autohealing policy for the instances in this managed instance group. [Deprecated] This method is deprecated. Use regionInstanceGroupManagers.patch instead.

      Args:
        request: (ComputeRegionInstanceGroupManagersSetAutoHealingPoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetAutoHealingPolicies')
    return self._RunMethod(config, request, global_params=global_params)