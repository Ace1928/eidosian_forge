from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListTraceSinksResponse(_messages.Message):
    """Result returned from `ListTraceSinks`.

  Fields:
    nextPageToken: A paginated response where more pages might be available
      has `next_page_token` set. To get the next set of results, call the same
      method again using the value of `next_page_token` as `page_token`.
    sinks: A list of sinks.
  """
    nextPageToken = _messages.StringField(1)
    sinks = _messages.MessageField('TraceSink', 2, repeated=True)