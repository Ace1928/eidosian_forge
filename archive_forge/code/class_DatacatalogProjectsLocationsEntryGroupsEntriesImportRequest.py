from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsEntriesImportRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsEntriesImportRequest object.

  Fields:
    googleCloudDatacatalogV1ImportEntriesRequest: A
      GoogleCloudDatacatalogV1ImportEntriesRequest resource to be passed as
      the request body.
    parent: Required. Target entry group for ingested entries.
  """
    googleCloudDatacatalogV1ImportEntriesRequest = _messages.MessageField('GoogleCloudDatacatalogV1ImportEntriesRequest', 1)
    parent = _messages.StringField(2, required=True)