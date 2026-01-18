from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryGroupsCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryGroupsCreateRequest object.

  Fields:
    entryGroupId: Required. EntryGroup identifier.
    googleCloudDataplexV1EntryGroup: A GoogleCloudDataplexV1EntryGroup
      resource to be passed as the request body.
    parent: Required. The resource name of the entryGroup, of the form:
      projects/{project_number}/locations/{location_id} where location_id
      refers to a GCP region.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    entryGroupId = _messages.StringField(1)
    googleCloudDataplexV1EntryGroup = _messages.MessageField('GoogleCloudDataplexV1EntryGroup', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)