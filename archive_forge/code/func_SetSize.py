from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def SetSize(self, request, global_params=None):
    """Sets the size for a specific node pool. The new size will be used for all replicas, including future replicas created by modifying NodePool.locations.

      Args:
        request: (SetNodePoolSizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetSize')
    return self._RunMethod(config, request, global_params=global_params)