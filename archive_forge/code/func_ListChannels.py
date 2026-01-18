import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
def ListChannels(self, request, global_params=None):
    """List active object change notification channels for this bucket.

      Args:
        request: (StorageBucketsListChannelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Channels) The response message.
      """
    config = self.GetMethodConfig('ListChannels')
    return self._RunMethod(config, request, global_params=global_params)