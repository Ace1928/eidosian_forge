from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
def EnableHyperthreading(self, request, global_params=None):
    """Perform enable hyperthreading operation on a single server.

      Args:
        request: (BaremetalsolutionProjectsLocationsInstancesEnableHyperthreadingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('EnableHyperthreading')
    return self._RunMethod(config, request, global_params=global_params)