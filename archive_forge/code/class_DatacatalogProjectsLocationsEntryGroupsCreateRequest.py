from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsCreateRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsCreateRequest object.

  Fields:
    entryGroupId: Required. The ID of the entry group to create. The ID must
      contain only letters (a-z, A-Z), numbers (0-9), underscores (_), and
      must start with a letter or underscore. The maximum size is 64 bytes
      when encoded in UTF-8.
    googleCloudDatacatalogV1EntryGroup: A GoogleCloudDatacatalogV1EntryGroup
      resource to be passed as the request body.
    parent: Required. The names of the project and location that the new entry
      group belongs to. Note: The entry group itself and its child resources
      might not be stored in the location specified in its name.
  """
    entryGroupId = _messages.StringField(1)
    googleCloudDatacatalogV1EntryGroup = _messages.MessageField('GoogleCloudDatacatalogV1EntryGroup', 2)
    parent = _messages.StringField(3, required=True)