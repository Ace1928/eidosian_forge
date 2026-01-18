from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListCrawledUrlsResponse(_messages.Message):
    """Response for the `ListCrawledUrls` method.

  Fields:
    crawledUrls: The list of CrawledUrls returned.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    crawledUrls = _messages.MessageField('CrawledUrl', 1, repeated=True)
    nextPageToken = _messages.StringField(2)