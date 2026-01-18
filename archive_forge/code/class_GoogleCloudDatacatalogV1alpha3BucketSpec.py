from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3BucketSpec(_messages.Message):
    """Configuration for a crawl bucket.

  Fields:
    bucket: The Google Cloud Storage bucket name.
  """
    bucket = _messages.StringField(1)