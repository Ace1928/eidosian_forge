from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def AddResourcePolicies(self, request, global_params=None):
    """Adds existing resource policies to a regional disk. You can only add one policy which will be applied to this disk for scheduling snapshot creation.

      Args:
        request: (ComputeRegionDisksAddResourcePoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('AddResourcePolicies')
    return self._RunMethod(config, request, global_params=global_params)