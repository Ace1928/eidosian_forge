from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2StorageConfig(_messages.Message):
    """Shared message indicating Cloud storage type.

  Fields:
    bigQueryOptions: BigQuery options.
    cloudStorageOptions: Cloud Storage options.
    datastoreOptions: Google Cloud Datastore options.
    hybridOptions: Hybrid inspection options.
    timespanConfig: Configuration of the timespan of the items to include in
      scanning.
  """
    bigQueryOptions = _messages.MessageField('GooglePrivacyDlpV2BigQueryOptions', 1)
    cloudStorageOptions = _messages.MessageField('GooglePrivacyDlpV2CloudStorageOptions', 2)
    datastoreOptions = _messages.MessageField('GooglePrivacyDlpV2DatastoreOptions', 3)
    hybridOptions = _messages.MessageField('GooglePrivacyDlpV2HybridOptions', 4)
    timespanConfig = _messages.MessageField('GooglePrivacyDlpV2TimespanConfig', 5)