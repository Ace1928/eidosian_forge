from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsEntriesCreateRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsEntriesCreateRequest object.

  Fields:
    entryId: Required. The ID of the entry to create. The ID must contain only
      letters (a-z, A-Z), numbers (0-9), and underscores (_). The maximum size
      is 64 bytes when encoded in UTF-8.
    googleCloudDatacatalogV1Entry: A GoogleCloudDatacatalogV1Entry resource to
      be passed as the request body.
    parent: Required. The name of the entry group this entry belongs to. Note:
      The entry itself and its child resources might not be stored in the
      location specified in its name.
  """
    entryId = _messages.StringField(1)
    googleCloudDatacatalogV1Entry = _messages.MessageField('GoogleCloudDatacatalogV1Entry', 2)
    parent = _messages.StringField(3, required=True)