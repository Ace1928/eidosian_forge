from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def SetManagement(self, request, global_params=None):
    """Sets the NodeManagement options for a node pool.

      Args:
        request: (SetNodePoolManagementRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetManagement')
    return self._RunMethod(config, request, global_params=global_params)