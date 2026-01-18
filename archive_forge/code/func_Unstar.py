from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
def Unstar(self, request, global_params=None):
    """Marks an Entry as NOT starred by the current user. Starring information is private to each user.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesUnstarRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1UnstarEntryResponse) The response message.
      """
    config = self.GetMethodConfig('Unstar')
    return self._RunMethod(config, request, global_params=global_params)