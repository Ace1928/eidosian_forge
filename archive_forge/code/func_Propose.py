from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.telcoautomation.v1 import telcoautomation_v1_messages as messages
def Propose(self, request, global_params=None):
    """Proposes a blueprint for approval of changes.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsProposeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Blueprint) The response message.
      """
    config = self.GetMethodConfig('Propose')
    return self._RunMethod(config, request, global_params=global_params)