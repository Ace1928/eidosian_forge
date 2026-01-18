from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchAllResourcesResponse(_messages.Message):
    """Search all resources response.

  Fields:
    nextPageToken: If there are more results than those appearing in this
      response, then `next_page_token` is included. To get the next set of
      results, call this method again using the value of `next_page_token` as
      `page_token`.
    results: A list of Resources that match the search query. It contains the
      resource standard metadata information.
  """
    nextPageToken = _messages.StringField(1)
    results = _messages.MessageField('ResourceSearchResult', 2, repeated=True)