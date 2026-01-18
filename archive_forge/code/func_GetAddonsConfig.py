from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetAddonsConfig(self, request, global_params=None):
    """Gets the add-ons config of an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetAddonsConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1AddonsConfig) The response message.
      """
    config = self.GetMethodConfig('GetAddonsConfig')
    return self._RunMethod(config, request, global_params=global_params)