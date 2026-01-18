from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSpokesResponse(_messages.Message):
    """The response for HubService.ListSpokes.

  Fields:
    nextPageToken: The token for the next page of the response. To see more
      results, use this value as the page_token for your next request. If this
      value is empty, there are no more results.
    spokes: The requested spokes.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    spokes = _messages.MessageField('Spoke', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)