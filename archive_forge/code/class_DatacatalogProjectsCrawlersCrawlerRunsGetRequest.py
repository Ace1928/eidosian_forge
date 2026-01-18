from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsCrawlersCrawlerRunsGetRequest(_messages.Message):
    """A DatacatalogProjectsCrawlersCrawlerRunsGetRequest object.

  Fields:
    name: Required. Resource name of the crawler run to retrieve. For example,
      "projects/project1/crawlers/crawler1/crawlerRuns/run1".
  """
    name = _messages.StringField(1, required=True)