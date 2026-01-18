from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
def CreateAndAttach(self, request, global_params=None):
    """Create a volume, allocate Luns and attach them to instances.

      Args:
        request: (BaremetalsolutionProjectsLocationsVolumesCreateAndAttachRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('CreateAndAttach')
    return self._RunMethod(config, request, global_params=global_params)