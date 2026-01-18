from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def ListServerCas(self, request, global_params=None):
    """Lists all of the trusted Certificate Authorities (CAs) for the specified instance. There can be up to three CAs listed: the CA that was used to sign the certificate that is currently in use, a CA that has been added but not yet used to sign a certificate, and a CA used to sign a certificate that has previously rotated out.

      Args:
        request: (SqlInstancesListServerCasRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstancesListServerCasResponse) The response message.
      """
    config = self.GetMethodConfig('ListServerCas')
    return self._RunMethod(config, request, global_params=global_params)