from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBatchesResponse(_messages.Message):
    """A list of batch workloads.

  Fields:
    batches: Output only. The batches from the specified collection.
    nextPageToken: A token, which can be sent as page_token to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    unreachable: Output only. List of Batches that could not be included in
      the response. Attempting to get one of these resources may indicate why
      it was not included in the list response.
  """
    batches = _messages.MessageField('Batch', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)