from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v1 import notebooks_v1_messages as messages
def GetInstanceHealth(self, request, global_params=None):
    """Checks whether a notebook instance is healthy.

      Args:
        request: (NotebooksProjectsLocationsInstancesGetInstanceHealthRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetInstanceHealthResponse) The response message.
      """
    config = self.GetMethodConfig('GetInstanceHealth')
    return self._RunMethod(config, request, global_params=global_params)