from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetManagedProtectionTier(self, request, global_params=None):
    """Sets the Cloud Armor Managed Protection (CAMP) tier of the project. To set PLUS or above the billing account of the project must be subscribed to Managed Protection Plus. See Subscribing to Managed Protection Plus for more information.

      Args:
        request: (ComputeProjectsSetManagedProtectionTierRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetManagedProtectionTier')
    return self._RunMethod(config, request, global_params=global_params)