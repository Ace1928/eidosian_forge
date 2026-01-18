from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsCrawlersCrawlerRunsListRequest(_messages.Message):
    """A DatacatalogProjectsCrawlersCrawlerRunsListRequest object.

  Fields:
    pageSize: The maximum number of items to return. The default value for
      this field is 10. The maximum value for this field is 1000.
    pageToken: The next_page_token value returned from a previous list
      request, if any.
    parent: Required. Resource name of the parent crawler resource. For
      example, "projects/project1/crawlers/crawler1".
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)