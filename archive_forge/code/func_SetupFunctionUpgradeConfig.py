from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudfunctions.v2 import cloudfunctions_v2_messages as messages
def SetupFunctionUpgradeConfig(self, request, global_params=None):
    """Creates a 2nd Gen copy of the function configuration based on the 1st Gen function with the given name. This is the first step of the multi step process to upgrade 1st Gen functions to 2nd Gen. Only 2nd Gen configuration is setup as part of this request and traffic continues to be served by 1st Gen.

      Args:
        request: (CloudfunctionsProjectsLocationsFunctionsSetupFunctionUpgradeConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetupFunctionUpgradeConfig')
    return self._RunMethod(config, request, global_params=global_params)