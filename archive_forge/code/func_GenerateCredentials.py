from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.connectgateway.v1alpha1 import connectgateway_v1alpha1_messages as messages
def GenerateCredentials(self, request, global_params=None):
    """GenerateCredentials provides connection information that allows a user to access the specified membership using Connect Gateway.

      Args:
        request: (ConnectgatewayProjectsLocationsMembershipsGenerateCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateCredentialsResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateCredentials')
    return self._RunMethod(config, request, global_params=global_params)