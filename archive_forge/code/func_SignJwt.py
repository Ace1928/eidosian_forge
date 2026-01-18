from __future__ import absolute_import
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.iamcredentials_apitools.iamcredentials_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
def SignJwt(self, request, global_params=None):
    """Signs a JWT using a service account's system-managed private key.

      Args:
        request: (IamcredentialsProjectsServiceAccountsSignJwtRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SignJwtResponse) The response message.
      """
    config = self.GetMethodConfig('SignJwt')
    return self._RunMethod(config, request, global_params=global_params)