from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsEntriesModifyEntryOverviewRequest(_messages.Message):
    """A
  DatacatalogProjectsLocationsEntryGroupsEntriesModifyEntryOverviewRequest
  object.

  Fields:
    googleCloudDatacatalogV1ModifyEntryOverviewRequest: A
      GoogleCloudDatacatalogV1ModifyEntryOverviewRequest resource to be passed
      as the request body.
    name: Required. The full resource name of the entry.
  """
    googleCloudDatacatalogV1ModifyEntryOverviewRequest = _messages.MessageField('GoogleCloudDatacatalogV1ModifyEntryOverviewRequest', 1)
    name = _messages.StringField(2, required=True)