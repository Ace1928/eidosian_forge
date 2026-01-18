from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def Upgrade(self, request, global_params=None):
    """Upgrades a NotebookRuntime.

      Args:
        request: (AiplatformProjectsLocationsNotebookRuntimesUpgradeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('Upgrade')
    return self._RunMethod(config, request, global_params=global_params)