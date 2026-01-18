from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudiot.v1 import cloudiot_v1_messages as messages
def BindDeviceToGateway(self, request, global_params=None):
    """Associates the device with the gateway.

      Args:
        request: (CloudiotProjectsLocationsRegistriesBindDeviceToGatewayRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BindDeviceToGatewayResponse) The response message.
      """
    config = self.GetMethodConfig('BindDeviceToGateway')
    return self._RunMethod(config, request, global_params=global_params)