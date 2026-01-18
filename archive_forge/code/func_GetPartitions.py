from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
def GetPartitions(self, request, global_params=None):
    """Returns the partition information for the requested topic.

      Args:
        request: (PubsubliteAdminProjectsLocationsTopicsGetPartitionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TopicPartitions) The response message.
      """
    config = self.GetMethodConfig('GetPartitions')
    return self._RunMethod(config, request, global_params=global_params)