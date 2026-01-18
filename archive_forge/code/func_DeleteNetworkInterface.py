from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
def DeleteNetworkInterface(self, request, global_params=None):
    """Deletes one network interface from an active instance. InstancesDeleteNetworkInterfaceRequest indicates: - instance from which to delete, using project+zone+resource_id fields; - network interface to be deleted, using network_interface_name field; Only VLAN interface deletion is supported for now.

      Args:
        request: (ComputeInstancesDeleteNetworkInterfaceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DeleteNetworkInterface')
    return self._RunMethod(config, request, global_params=global_params)