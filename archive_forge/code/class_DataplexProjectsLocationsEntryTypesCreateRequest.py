from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryTypesCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryTypesCreateRequest object.

  Fields:
    entryTypeId: Required. EntryType identifier.
    googleCloudDataplexV1EntryType: A GoogleCloudDataplexV1EntryType resource
      to be passed as the request body.
    parent: Required. The resource name of the EntryType, of the form:
      projects/{project_number}/locations/{location_id} where location_id
      refers to a GCP region.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    entryTypeId = _messages.StringField(1)
    googleCloudDataplexV1EntryType = _messages.MessageField('GoogleCloudDataplexV1EntryType', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)