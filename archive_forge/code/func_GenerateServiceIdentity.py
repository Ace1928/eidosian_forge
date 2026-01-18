from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v1beta1 import serviceusage_v1beta1_messages as messages
def GenerateServiceIdentity(self, request, global_params=None):
    """Generate service identity for service.

      Args:
        request: (ServiceusageServicesGenerateServiceIdentityRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('GenerateServiceIdentity')
    return self._RunMethod(config, request, global_params=global_params)