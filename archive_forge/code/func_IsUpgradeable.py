from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v1 import notebooks_v1_messages as messages
def IsUpgradeable(self, request, global_params=None):
    """Checks whether a notebook instance is upgradable.

      Args:
        request: (NotebooksProjectsLocationsInstancesIsUpgradeableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IsInstanceUpgradeableResponse) The response message.
      """
    config = self.GetMethodConfig('IsUpgradeable')
    return self._RunMethod(config, request, global_params=global_params)