from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1ListEntryGroupsResponse(_messages.Message):
    """Response message for ListEntryGroups.

  Fields:
    entryGroups: EntryGroup details.
    nextPageToken: Token to retrieve the next page of results. It is set to
      empty if no items remain in results.
  """
    entryGroups = _messages.MessageField('GoogleCloudDatacatalogV1beta1EntryGroup', 1, repeated=True)
    nextPageToken = _messages.StringField(2)