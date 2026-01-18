from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
def DetachLun(self, request, global_params=None):
    """Detach LUN from Instance.

      Args:
        request: (BaremetalsolutionProjectsLocationsInstancesDetachLunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DetachLun')
    return self._RunMethod(config, request, global_params=global_params)