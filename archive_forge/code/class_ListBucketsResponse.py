from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBucketsResponse(_messages.Message):
    """The response from ListBuckets.

  Fields:
    buckets: A list of buckets.
    nextPageToken: If there might be more results than appear in this
      response, then nextPageToken is included. To get the next set of
      results, call the same method again using the value of nextPageToken as
      pageToken.
  """
    buckets = _messages.MessageField('LogBucket', 1, repeated=True)
    nextPageToken = _messages.StringField(2)