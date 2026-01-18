from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudfunctions.v2 import cloudfunctions_v2_messages as messages
def RollbackFunctionUpgradeTraffic(self, request, global_params=None):
    """Reverts the traffic target of a function from the 2nd Gen copy to the original 1st Gen function. After this operation, all new traffic would be served by the 1st Gen.

      Args:
        request: (CloudfunctionsProjectsLocationsFunctionsRollbackFunctionUpgradeTrafficRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('RollbackFunctionUpgradeTraffic')
    return self._RunMethod(config, request, global_params=global_params)