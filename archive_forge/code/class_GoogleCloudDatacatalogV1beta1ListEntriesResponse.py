from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1ListEntriesResponse(_messages.Message):
    """Response message for ListEntries.

  Fields:
    entries: Entry details.
    nextPageToken: Token to retrieve the next page of results. It is set to
      empty if no items remain in results.
  """
    entries = _messages.MessageField('GoogleCloudDatacatalogV1beta1Entry', 1, repeated=True)
    nextPageToken = _messages.StringField(2)