from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
def ModifyEntryContacts(self, request, global_params=None):
    """Modifies contacts, part of the business context of an Entry. To call this method, you must have the `datacatalog.entries.updateContacts` IAM permission on the corresponding project.

      Args:
        request: (DatacatalogProjectsLocationsEntryGroupsEntriesModifyEntryContactsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Contacts) The response message.
      """
    config = self.GetMethodConfig('ModifyEntryContacts')
    return self._RunMethod(config, request, global_params=global_params)