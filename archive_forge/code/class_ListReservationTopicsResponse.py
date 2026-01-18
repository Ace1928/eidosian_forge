from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListReservationTopicsResponse(_messages.Message):
    """Response for ListReservationTopics.

  Fields:
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page of results. If this field is omitted, there are no more
      results.
    topics: The names of topics attached to the reservation. The order of the
      topics is unspecified.
  """
    nextPageToken = _messages.StringField(1)
    topics = _messages.StringField(2, repeated=True)