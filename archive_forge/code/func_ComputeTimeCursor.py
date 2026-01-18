from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
def ComputeTimeCursor(self, request, global_params=None):
    """Compute the corresponding cursor for a publish or event time in a topic partition.

      Args:
        request: (PubsubliteTopicStatsProjectsLocationsTopicsComputeTimeCursorRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ComputeTimeCursorResponse) The response message.
      """
    config = self.GetMethodConfig('ComputeTimeCursor')
    return self._RunMethod(config, request, global_params=global_params)