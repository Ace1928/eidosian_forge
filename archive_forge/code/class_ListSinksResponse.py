from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSinksResponse(_messages.Message):
    """Result returned from ListSinks.

  Fields:
    nextPageToken: If there might be more results than appear in this
      response, then nextPageToken is included. To get the next set of
      results, call the same method again using the value of nextPageToken as
      pageToken.
    sinks: A list of sinks.
  """
    nextPageToken = _messages.StringField(1)
    sinks = _messages.MessageField('LogSink', 2, repeated=True)