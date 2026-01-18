from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def DisableXpnResource(self, request, global_params=None):
    """Disable a service resource (also known as service project) associated with this host project.

      Args:
        request: (ComputeProjectsDisableXpnResourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DisableXpnResource')
    return self._RunMethod(config, request, global_params=global_params)