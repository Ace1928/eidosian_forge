from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.netapp.v1 import netapp_v1_messages as messages
def ReverseDirection(self, request, global_params=None):
    """Reverses direction of replication. Source becomes destination and destination becomes source.

      Args:
        request: (NetappProjectsLocationsVolumesReplicationsReverseDirectionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ReverseDirection')
    return self._RunMethod(config, request, global_params=global_params)