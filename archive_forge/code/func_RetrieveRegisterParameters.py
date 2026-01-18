from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.domains.v1alpha2 import domains_v1alpha2_messages as messages
def RetrieveRegisterParameters(self, request, global_params=None):
    """Gets parameters needed to register a new domain name, including price and up-to-date availability. Use the returned values to call `RegisterDomain`.

      Args:
        request: (DomainsProjectsLocationsRegistrationsRetrieveRegisterParametersRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RetrieveRegisterParametersResponse) The response message.
      """
    config = self.GetMethodConfig('RetrieveRegisterParameters')
    return self._RunMethod(config, request, global_params=global_params)