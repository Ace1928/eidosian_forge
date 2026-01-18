from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListLinksResponse(_messages.Message):
    """The response from ListLinks.

  Fields:
    links: A list of links.
    nextPageToken: If there might be more results than those appearing in this
      response, then nextPageToken is included. To get the next set of
      results, call the same method again using the value of nextPageToken as
      pageToken.
  """
    links = _messages.MessageField('Link', 1, repeated=True)
    nextPageToken = _messages.StringField(2)