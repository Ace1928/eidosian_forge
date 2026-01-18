from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetTarget(self, request, global_params=None):
    """Changes target URL for the GlobalForwardingRule resource. The new target should be of the same type as the old target.

      Args:
        request: (ComputeGlobalForwardingRulesSetTargetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetTarget')
    return self._RunMethod(config, request, global_params=global_params)