from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryTypesDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryTypesDeleteRequest object.

  Fields:
    etag: Optional. If the client provided etag value does not match the
      current etag value, the DeleteEntryTypeRequest method returns an ABORTED
      error response
    name: Required. The resource name of the EntryType: projects/{project_numb
      er}/locations/{location_id}/entryTypes/{entry_type_id}.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)