from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1SearchEntriesResponse(_messages.Message):
    """A GoogleCloudDataplexV1SearchEntriesResponse object.

  Fields:
    nextPageToken: Pagination token.
    results: The results matching the search query.
    totalSize: The estimated total number of matching entries. Not guaranteed
      to be accurate.
    unreachable: Unreachable locations. Search results don't include data from
      those locations.
  """
    nextPageToken = _messages.StringField(1)
    results = _messages.MessageField('GoogleCloudDataplexV1SearchEntriesResult', 2, repeated=True)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    unreachable = _messages.StringField(4, repeated=True)