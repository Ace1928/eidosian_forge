from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def SetDefaultServiceAccount(self, request, global_params=None):
    """Sets the default service account of the project. The default service account is used when a VM instance is created with the service account email address set to "default".

      Args:
        request: (ComputeProjectsSetDefaultServiceAccountRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetDefaultServiceAccount')
    return self._RunMethod(config, request, global_params=global_params)