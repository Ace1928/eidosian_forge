from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3ListCrawlerRunsResponse(_messages.Message):
    """Response to listing the runs from a crawler.

  Fields:
    crawlerRuns: The crawler runs from this crawler, it includes both
      currently running and completed runs.
    nextPageToken: Token to retrieve the next page of results or empty if
      there are no more results in the list.
  """
    crawlerRuns = _messages.MessageField('GoogleCloudDatacatalogV1alpha3CrawlerRun', 1, repeated=True)
    nextPageToken = _messages.StringField(2)