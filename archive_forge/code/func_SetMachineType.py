from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v1 import notebooks_v1_messages as messages
def SetMachineType(self, request, global_params=None):
    """Updates the machine type of a single Instance.

      Args:
        request: (NotebooksProjectsLocationsInstancesSetMachineTypeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetMachineType')
    return self._RunMethod(config, request, global_params=global_params)