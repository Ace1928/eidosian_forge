from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def Credit(self, request, global_params=None):
    """Credits the account balance for the developer.

      Args:
        request: (ApigeeOrganizationsDevelopersBalanceCreditRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperBalance) The response message.
      """
    config = self.GetMethodConfig('Credit')
    return self._RunMethod(config, request, global_params=global_params)