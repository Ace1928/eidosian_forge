from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatasetTemplate(_messages.Message):
    """Dataset template used for dynamic dataset creation.

  Fields:
    datasetIdPrefix: If supplied, every created dataset will have its name
      prefixed by the provided value. The prefix and name will be separated by
      an underscore. i.e. _.
    kmsKeyName: Describes the Cloud KMS encryption key that will be used to
      protect destination BigQuery table. The BigQuery Service Account
      associated with your project requires access to this encryption key.
      i.e. projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoK
      eys/{cryptoKey}. See https://cloud.google.com/bigquery/docs/customer-
      managed-encryption for more information.
    location: Required. The geographic location where the dataset should
      reside. See https://cloud.google.com/bigquery/docs/locations for
      supported locations.
  """
    datasetIdPrefix = _messages.StringField(1)
    kmsKeyName = _messages.StringField(2)
    location = _messages.StringField(3)