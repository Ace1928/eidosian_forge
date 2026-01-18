from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3ListCrawlersResponse(_messages.Message):
    """Response message for `ListCrawlers` API.

  Fields:
    crawlers: List of crawlers.
    nextPageToken: Token to retrieve the next page of results or empty if
      there are no more results in the list.
  """
    crawlers = _messages.MessageField('GoogleCloudDatacatalogV1alpha3Crawler', 1, repeated=True)
    nextPageToken = _messages.StringField(2)