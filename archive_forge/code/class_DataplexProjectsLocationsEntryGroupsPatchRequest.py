from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryGroupsPatchRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryGroupsPatchRequest object.

  Fields:
    googleCloudDataplexV1EntryGroup: A GoogleCloudDataplexV1EntryGroup
      resource to be passed as the request body.
    name: Output only. The relative resource name of the EntryGroup, of the
      form: projects/{project_number}/locations/{location_id}/entryGroups/{ent
      ry_group_id}.
    updateMask: Required. Mask of fields to update.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1EntryGroup = _messages.MessageField('GoogleCloudDataplexV1EntryGroup', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)