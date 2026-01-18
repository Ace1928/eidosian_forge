from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3Crawler(_messages.Message):
    """Crawler Metadata.

  Fields:
    config: Required. The configuration of the crawler.
    description: The description of the crawler.
    displayName: Required. The display name of the crawler.
    name: Output only. The resource name of the crawler. Must be empty when
      creating a crawler. For example, "projects/a/crawlers/b".
  """
    config = _messages.MessageField('GoogleCloudDatacatalogV1alpha3CrawlerConfig', 1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)