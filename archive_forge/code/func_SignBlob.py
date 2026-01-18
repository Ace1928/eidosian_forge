from __future__ import absolute_import
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.iamcredentials_apitools.iamcredentials_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
def SignBlob(self, request, global_params=None):
    """Signs a blob using a service account's system-managed private key.

      Args:
        request: (IamcredentialsProjectsServiceAccountsSignBlobRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SignBlobResponse) The response message.
      """
    config = self.GetMethodConfig('SignBlob')
    return self._RunMethod(config, request, global_params=global_params)