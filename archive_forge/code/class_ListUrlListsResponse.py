from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUrlListsResponse(_messages.Message):
    """Response returned by the ListUrlLists method.

  Fields:
    nextPageToken: If there might be more results than those appearing in this
      response, then `next_page_token` is included. To get the next set of
      results, call this method again using the value of `next_page_token` as
      `page_token`.
    unreachable: Locations that could not be reached.
    urlLists: List of UrlList resources.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    urlLists = _messages.MessageField('UrlList', 3, repeated=True)