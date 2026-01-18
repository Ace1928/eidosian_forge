from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1beta import networkconnectivity_v1beta_messages as messages
def ListSpokes(self, request, global_params=None):
    """Lists the Network Connectivity Center spokes associated with a specified hub and location. The list includes both spokes that are attached to the hub and spokes that have been proposed but not yet accepted.

      Args:
        request: (NetworkconnectivityProjectsLocationsGlobalHubsListSpokesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaListHubSpokesResponse) The response message.
      """
    config = self.GetMethodConfig('ListSpokes')
    return self._RunMethod(config, request, global_params=global_params)