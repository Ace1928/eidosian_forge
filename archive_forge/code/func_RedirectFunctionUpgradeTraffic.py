from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudfunctions.v2 import cloudfunctions_v2_messages as messages
def RedirectFunctionUpgradeTraffic(self, request, global_params=None):
    """Changes the traffic target of a function from the original 1st Gen function to the 2nd Gen copy. This is the second step of the multi step process to upgrade 1st Gen functions to 2nd Gen. After this operation, all new traffic will be served by 2nd Gen copy.

      Args:
        request: (CloudfunctionsProjectsLocationsFunctionsRedirectFunctionUpgradeTrafficRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('RedirectFunctionUpgradeTraffic')
    return self._RunMethod(config, request, global_params=global_params)