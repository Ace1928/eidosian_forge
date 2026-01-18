from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
def ListReferences(self, request, global_params=None):
    """Lists references of an address group.

      Args:
        request: (NetworksecurityProjectsLocationsAddressGroupsListReferencesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAddressGroupReferencesResponse) The response message.
      """
    config = self.GetMethodConfig('ListReferences')
    return self._RunMethod(config, request, global_params=global_params)