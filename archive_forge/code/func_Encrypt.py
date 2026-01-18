from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.kms_apitools.cloudkms_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
def Encrypt(self, request, global_params=None):
    """Encrypts data, so that it can only be recovered by a call to Decrypt.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysEncryptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EncryptResponse) The response message.
      """
    config = self.GetMethodConfig('Encrypt')
    return self._RunMethod(config, request, global_params=global_params)