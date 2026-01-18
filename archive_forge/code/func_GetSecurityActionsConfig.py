from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetSecurityActionsConfig(self, request, global_params=None):
    """GetSecurityActionConfig returns the current SecurityActions configuration.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetSecurityActionsConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityActionsConfig) The response message.
      """
    config = self.GetMethodConfig('GetSecurityActionsConfig')
    return self._RunMethod(config, request, global_params=global_params)