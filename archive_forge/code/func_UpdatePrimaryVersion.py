from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.kms_apitools.cloudkms_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
def UpdatePrimaryVersion(self, request, global_params=None):
    """Update the version of a CryptoKey that will be used in Encrypt.

      Args:
        request: (CloudkmsProjectsLocationsKeyRingsCryptoKeysUpdatePrimaryVersionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CryptoKey) The response message.
      """
    config = self.GetMethodConfig('UpdatePrimaryVersion')
    return self._RunMethod(config, request, global_params=global_params)