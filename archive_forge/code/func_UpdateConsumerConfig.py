from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
def UpdateConsumerConfig(self, request, global_params=None):
    """Service producers use this method to update the configuration of their connection including the import/export of custom routes and subnetwork routes with public IP.

      Args:
        request: (ServicenetworkingServicesProjectsGlobalNetworksUpdateConsumerConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpdateConsumerConfig')
    return self._RunMethod(config, request, global_params=global_params)