from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def UpdateSecurityActionsConfig(self, request, global_params=None):
    """UpdateSecurityActionConfig updates the current SecurityActions configuration. This method is used to enable/disable the feature at the environment level.

      Args:
        request: (ApigeeOrganizationsEnvironmentsUpdateSecurityActionsConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityActionsConfig) The response message.
      """
    config = self.GetMethodConfig('UpdateSecurityActionsConfig')
    return self._RunMethod(config, request, global_params=global_params)