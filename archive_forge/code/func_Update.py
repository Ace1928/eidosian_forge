import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
def Update(self, request, global_params=None):
    """Updates the state of an HMAC key.

      Args:
        request: (StorageProjectsHmacKeysUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (HmacKeyMetadata) The response message.
      """
    config = self.GetMethodConfig('Update')
    return self._RunMethod(config, request, global_params=global_params)