from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
def Resolve(self, request, global_params=None):
    """Resolves connections details for a given connector. An internal method called by a connector to find connections to connect to.

      Args:
        request: (BeyondcorpProjectsLocationsConnectionsResolveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResolveConnectionsResponse) The response message.
      """
    config = self.GetMethodConfig('Resolve')
    return self._RunMethod(config, request, global_params=global_params)