from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.domains.v1alpha2 import domains_v1alpha2_messages as messages
def RetrieveAuthorizationCode(self, request, global_params=None):
    """Gets the authorization code of the `Registration` for the purpose of transferring the domain to another registrar. You can call this method only after 60 days have elapsed since the initial domain registration.

      Args:
        request: (DomainsProjectsLocationsRegistrationsRetrieveAuthorizationCodeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuthorizationCode) The response message.
      """
    config = self.GetMethodConfig('RetrieveAuthorizationCode')
    return self._RunMethod(config, request, global_params=global_params)