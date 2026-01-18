import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
def Stop(self, request, global_params=None):
    """Stop watching resources through this channel.

      Args:
        request: (Channel) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StorageChannelsStopResponse) The response message.
      """
    config = self.GetMethodConfig('Stop')
    return self._RunMethod(config, request, global_params=global_params)