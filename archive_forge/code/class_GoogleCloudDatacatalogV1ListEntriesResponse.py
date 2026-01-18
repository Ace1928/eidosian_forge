from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ListEntriesResponse(_messages.Message):
    """Response message for ListEntries.

  Fields:
    entries: Entry details.
    nextPageToken: Pagination token of the next results page. Empty if there
      are no more items in results.
  """
    entries = _messages.MessageField('GoogleCloudDatacatalogV1Entry', 1, repeated=True)
    nextPageToken = _messages.StringField(2)