import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
def WatchAll(self, request, global_params=None):
    """Watch for changes on all objects in a bucket.

      Args:
        request: (StorageObjectsWatchAllRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Channel) The response message.
      """
    config = self.GetMethodConfig('WatchAll')
    return self._RunMethod(config, request, global_params=global_params)