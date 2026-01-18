from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def SetMaintenancePolicy(self, request, global_params=None):
    """Sets the maintenance policy for a cluster.

      Args:
        request: (SetMaintenancePolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetMaintenancePolicy')
    return self._RunMethod(config, request, global_params=global_params)