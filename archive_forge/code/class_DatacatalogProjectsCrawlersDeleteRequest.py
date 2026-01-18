from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsCrawlersDeleteRequest(_messages.Message):
    """A DatacatalogProjectsCrawlersDeleteRequest object.

  Fields:
    name: Required. The resource name of the crawler. For example,
      "projects/bar/crawlers/foo".
  """
    name = _messages.StringField(1, required=True)