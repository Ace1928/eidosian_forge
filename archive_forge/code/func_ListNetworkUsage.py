from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
def ListNetworkUsage(self, request, global_params=None):
    """List all Networks (and used IPs for each Network) in the vendor account associated with the specified project.

      Args:
        request: (BaremetalsolutionProjectsLocationsNetworksListNetworkUsageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNetworkUsageResponse) The response message.
      """
    config = self.GetMethodConfig('ListNetworkUsage')
    return self._RunMethod(config, request, global_params=global_params)