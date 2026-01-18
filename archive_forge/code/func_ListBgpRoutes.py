from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListBgpRoutes(self, request, global_params=None):
    """Retrieves a list of router bgp routes available to the specified project.

      Args:
        request: (ComputeRoutersListBgpRoutesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RoutersListBgpRoutes) The response message.
      """
    config = self.GetMethodConfig('ListBgpRoutes')
    return self._RunMethod(config, request, global_params=global_params)