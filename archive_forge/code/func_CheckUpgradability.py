from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v2 import notebooks_v2_messages as messages
def CheckUpgradability(self, request, global_params=None):
    """Checks whether a notebook instance is upgradable.

      Args:
        request: (NotebooksProjectsLocationsInstancesCheckUpgradabilityRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckInstanceUpgradabilityResponse) The response message.
      """
    config = self.GetMethodConfig('CheckUpgradability')
    return self._RunMethod(config, request, global_params=global_params)