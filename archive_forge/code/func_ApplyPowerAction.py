from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.marketplacesolutions.v1alpha1 import marketplacesolutions_v1alpha1_messages as messages
def ApplyPowerAction(self, request, global_params=None):
    """Performs one of several power-related actions on an instance.

      Args:
        request: (MarketplacesolutionsProjectsLocationsPowerInstancesApplyPowerActionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ApplyPowerAction')
    return self._RunMethod(config, request, global_params=global_params)