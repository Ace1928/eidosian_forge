from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privateca.v1 import privateca_v1_messages as messages
def FetchCaCerts(self, request, global_params=None):
    """FetchCaCerts returns the current trust anchor for the CaPool. This will include CA certificate chains for all certificate authorities in the ENABLED, DISABLED, or STAGED states.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsFetchCaCertsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchCaCertsResponse) The response message.
      """
    config = self.GetMethodConfig('FetchCaCerts')
    return self._RunMethod(config, request, global_params=global_params)