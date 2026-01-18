from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetBalance(self, request, global_params=None):
    """Gets the account balance for the developer.

      Args:
        request: (ApigeeOrganizationsDevelopersGetBalanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperBalance) The response message.
      """
    config = self.GetMethodConfig('GetBalance')
    return self._RunMethod(config, request, global_params=global_params)