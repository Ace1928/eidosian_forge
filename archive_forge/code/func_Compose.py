import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
def Compose(self, request, global_params=None):
    """Concatenates a list of existing objects into a new object in the same bucket.

      Args:
        request: (StorageObjectsComposeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Object) The response message.
      """
    config = self.GetMethodConfig('Compose')
    return self._RunMethod(config, request, global_params=global_params)