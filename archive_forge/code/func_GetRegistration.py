from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
def GetRegistration(self, request, global_params=None):
    """Get a URL that a customer should use to initiate an OAuth flow on an external source provider. This API is experimental.

      Args:
        request: (CloudbuildOauthGetRegistrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OAuthRegistrationURI) The response message.
      """
    config = self.GetMethodConfig('GetRegistration')
    return self._RunMethod(config, request, global_params=global_params)