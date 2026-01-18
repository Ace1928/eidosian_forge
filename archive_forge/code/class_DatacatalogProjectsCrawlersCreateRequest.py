from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsCrawlersCreateRequest(_messages.Message):
    """A DatacatalogProjectsCrawlersCreateRequest object.

  Fields:
    crawlerId: Required. The id of the crawler to create.
    googleCloudDatacatalogV1alpha3Crawler: A
      GoogleCloudDatacatalogV1alpha3Crawler resource to be passed as the
      request body.
    parent: Required. The name of the project this crawler is in. Example:
      "projects/foo".
  """
    crawlerId = _messages.StringField(1)
    googleCloudDatacatalogV1alpha3Crawler = _messages.MessageField('GoogleCloudDatacatalogV1alpha3Crawler', 2)
    parent = _messages.StringField(3, required=True)