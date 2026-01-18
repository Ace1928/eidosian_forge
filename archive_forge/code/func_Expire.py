from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def Expire(self, request, global_params=None):
    """Expires an API product subscription immediately.

      Args:
        request: (ApigeeOrganizationsDevelopersSubscriptionsExpireRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperSubscription) The response message.
      """
    config = self.GetMethodConfig('Expire')
    return self._RunMethod(config, request, global_params=global_params)