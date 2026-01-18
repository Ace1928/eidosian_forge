from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudnumberregistry.v1alpha import cloudnumberregistry_v1alpha_messages as messages
def SearchRegistry(self, request, global_params=None):
    """Search registry nodes in a given registry book.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksSearchRegistryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchRegistryResponse) The response message.
      """
    config = self.GetMethodConfig('SearchRegistry')
    return self._RunMethod(config, request, global_params=global_params)