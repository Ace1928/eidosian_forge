import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
def Patch(self, request, global_params=None):
    """Patches an object's metadata.

      Args:
        request: (StorageObjectsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Object) The response message.
      """
    config = self.GetMethodConfig('Patch')
    return self._RunMethod(config, request, global_params=global_params)