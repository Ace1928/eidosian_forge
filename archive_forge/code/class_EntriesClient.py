from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import util
class EntriesClient(object):
    """Cloud Datacatalog entries client."""

    def __init__(self):
        self.client = util.GetClientInstance()
        self.messages = util.GetMessagesModule()
        self.entry_lookup_service = self.client.entries
        self.entry_service = self.client.projects_locations_entryGroups_entries

    def Lookup(self, resource):
        request = ParseResourceIntoLookupRequest(resource, self.messages.DatacatalogEntriesLookupRequest())
        return self.entry_lookup_service.Lookup(request)

    def Get(self, resource):
        request = self.messages.DatacatalogProjectsLocationsEntryGroupsEntriesGetRequest(name=resource.RelativeName())
        return self.entry_service.Get(request)