from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
def GetVpcServiceControls(self, request, global_params=None):
    """Consumers use this method to find out the state of VPC Service Controls. The controls could be enabled or disabled for a connection.

      Args:
        request: (ServicenetworkingServicesProjectsGlobalNetworksGetVpcServiceControlsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpcServiceControls) The response message.
      """
    config = self.GetMethodConfig('GetVpcServiceControls')
    return self._RunMethod(config, request, global_params=global_params)