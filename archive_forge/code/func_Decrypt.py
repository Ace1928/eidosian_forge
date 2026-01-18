from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.kms_apitools.cloudkms_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
def Decrypt(self, request, global_params=None):
    """Decrypts data that was protected by Encrypt.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysDecryptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DecryptResponse) The response message.
      """
    config = self.GetMethodConfig('Decrypt')
    return self._RunMethod(config, request, global_params=global_params)