from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
def DisableInteractiveSerialConsole(self, request, global_params=None):
    """Disable the interactive serial console feature on an instance.

      Args:
        request: (BaremetalsolutionProjectsLocationsInstancesDisableInteractiveSerialConsoleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DisableInteractiveSerialConsole')
    return self._RunMethod(config, request, global_params=global_params)