from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
def DisableHyperthreading(self, request, global_params=None):
    """Perform disable hyperthreading operation on a single server.

      Args:
        request: (BaremetalsolutionProjectsLocationsInstancesDisableHyperthreadingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DisableHyperthreading')
    return self._RunMethod(config, request, global_params=global_params)