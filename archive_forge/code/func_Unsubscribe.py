from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def Unsubscribe(self, request, global_params=None):
    """Deletes a subscription for the environment's Pub/Sub topic.

      Args:
        request: (ApigeeOrganizationsEnvironmentsUnsubscribeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
    config = self.GetMethodConfig('Unsubscribe')
    return self._RunMethod(config, request, global_params=global_params)