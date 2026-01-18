from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privateca.v1 import privateca_v1_messages as messages
def Fetch(self, request, global_params=None):
    """Fetch a certificate signing request (CSR) from a CertificateAuthority that is in state AWAITING_USER_ACTIVATION and is of type SUBORDINATE. The CSR must then be signed by the desired parent Certificate Authority, which could be another CertificateAuthority resource, or could be an on-prem certificate authority. See also ActivateCertificateAuthority.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesFetchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchCertificateAuthorityCsrResponse) The response message.
      """
    config = self.GetMethodConfig('Fetch')
    return self._RunMethod(config, request, global_params=global_params)