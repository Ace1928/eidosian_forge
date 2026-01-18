from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsEntriesModifyEntryContactsRequest(_messages.Message):
    """A
  DatacatalogProjectsLocationsEntryGroupsEntriesModifyEntryContactsRequest
  object.

  Fields:
    googleCloudDatacatalogV1ModifyEntryContactsRequest: A
      GoogleCloudDatacatalogV1ModifyEntryContactsRequest resource to be passed
      as the request body.
    name: Required. The full resource name of the entry.
  """
    googleCloudDatacatalogV1ModifyEntryContactsRequest = _messages.MessageField('GoogleCloudDatacatalogV1ModifyEntryContactsRequest', 1)
    name = _messages.StringField(2, required=True)