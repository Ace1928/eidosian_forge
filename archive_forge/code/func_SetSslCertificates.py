from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetSslCertificates(self, request, global_params=None):
    """Changes SslCertificates for TargetSslProxy.

      Args:
        request: (ComputeTargetSslProxiesSetSslCertificatesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetSslCertificates')
    return self._RunMethod(config, request, global_params=global_params)