from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privateca.v1 import privateca_v1_messages as messages
def Revoke(self, request, global_params=None):
    """Revoke a Certificate.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificatesRevokeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Certificate) The response message.
      """
    config = self.GetMethodConfig('Revoke')
    return self._RunMethod(config, request, global_params=global_params)