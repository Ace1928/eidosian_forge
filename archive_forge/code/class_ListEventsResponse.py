from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListEventsResponse(_messages.Message):
    """Contains a set of requested error events.

  Fields:
    errorEvents: The error events which match the given request.
    nextPageToken: If non-empty, more results are available. Pass this token,
      along with the same query parameters as the first request, to view the
      next page of results.
    timeRangeBegin: The timestamp specifies the start time to which the
      request was restricted.
  """
    errorEvents = _messages.MessageField('ErrorEvent', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    timeRangeBegin = _messages.StringField(3)