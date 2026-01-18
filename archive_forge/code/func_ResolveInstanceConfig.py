from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
def ResolveInstanceConfig(self, request, global_params=None):
    """Gets instance configuration for a given connector. An internal method called by a connector to get its container config.

      Args:
        request: (BeyondcorpProjectsLocationsConnectorsResolveInstanceConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResolveInstanceConfigResponse) The response message.
      """
    config = self.GetMethodConfig('ResolveInstanceConfig')
    return self._RunMethod(config, request, global_params=global_params)