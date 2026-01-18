from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryTypesGetRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryTypesGetRequest object.

  Fields:
    name: Required. The resource name of the EntryType: projects/{project_numb
      er}/locations/{location_id}/entryTypes/{entry_type_id}.
  """
    name = _messages.StringField(1, required=True)