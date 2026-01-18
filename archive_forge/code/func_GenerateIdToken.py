from __future__ import absolute_import
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.iamcredentials_apitools.iamcredentials_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
def GenerateIdToken(self, request, global_params=None):
    """Generates an OpenID Connect ID token for a service account.

      Args:
        request: (IamcredentialsProjectsServiceAccountsGenerateIdTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateIdTokenResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateIdToken')
    return self._RunMethod(config, request, global_params=global_params)