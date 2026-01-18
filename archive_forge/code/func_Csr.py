from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def Csr(self, request, global_params=None):
    """Generates a PKCS #10 Certificate Signing Request for the private key in an alias.

      Args:
        request: (ApigeeOrganizationsEnvironmentsKeystoresAliasesCsrRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
    config = self.GetMethodConfig('Csr')
    return self._RunMethod(config, request, global_params=global_params)