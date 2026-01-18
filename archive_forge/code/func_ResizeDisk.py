from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v2 import notebooks_v2_messages as messages
def ResizeDisk(self, request, global_params=None):
    """Resize a notebook instance disk to a higher capacity.

      Args:
        request: (NotebooksProjectsLocationsInstancesResizeDiskRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ResizeDisk')
    return self._RunMethod(config, request, global_params=global_params)