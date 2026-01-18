from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryGroupsDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryGroupsDeleteRequest object.

  Fields:
    etag: Optional. If the client provided etag value does not match the
      current etag value, the DeleteEntryGroupRequest method returns an
      ABORTED error response
    name: Required. The resource name of the EntryGroup: projects/{project_num
      ber}/locations/{location_id}/entryGroups/{entry_group_id}.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)