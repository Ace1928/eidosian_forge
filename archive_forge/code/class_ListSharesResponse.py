from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSharesResponse(_messages.Message):
    """ListSharesResponse is the result of ListSharesRequest.

  Fields:
    nextPageToken: The token you can use to retrieve the next page of results.
      Not returned if there are no more results in the list.
    shares: A list of shares in the project for the specified instance.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    shares = _messages.MessageField('Share', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)