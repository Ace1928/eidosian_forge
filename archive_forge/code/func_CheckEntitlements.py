from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudcommerceconsumerprocurement.v1alpha1 import cloudcommerceconsumerprocurement_v1alpha1_messages as messages
def CheckEntitlements(self, request, global_params=None):
    """Returns all active entitlements based on project and service type in its request.

      Args:
        request: (CloudcommerceconsumerprocurementProjectsCheckEntitlementsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1CheckEntitlementsResponse) The response message.
      """
    config = self.GetMethodConfig('CheckEntitlements')
    return self._RunMethod(config, request, global_params=global_params)