from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListQueuesResponse(_messages.Message):
    """Response message for ListQueues.

  Fields:
    nextPageToken: A token to retrieve next page of results. To return the
      next page of results, call ListQueues with this value as the page_token.
      If the next_page_token is empty, there are no more results. The page
      token is valid for only 2 hours.
    queues: The list of queues.
  """
    nextPageToken = _messages.StringField(1)
    queues = _messages.MessageField('Queue', 2, repeated=True)