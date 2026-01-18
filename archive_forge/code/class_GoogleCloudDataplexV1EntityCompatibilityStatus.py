from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EntityCompatibilityStatus(_messages.Message):
    """Provides compatibility information for various metadata stores.

  Fields:
    bigquery: Output only. Whether this entity is compatible with BigQuery.
    hiveMetastore: Output only. Whether this entity is compatible with Hive
      Metastore.
  """
    bigquery = _messages.MessageField('GoogleCloudDataplexV1EntityCompatibilityStatusCompatibility', 1)
    hiveMetastore = _messages.MessageField('GoogleCloudDataplexV1EntityCompatibilityStatusCompatibility', 2)