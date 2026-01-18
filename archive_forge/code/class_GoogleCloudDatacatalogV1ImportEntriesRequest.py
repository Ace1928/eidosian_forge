from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ImportEntriesRequest(_messages.Message):
    """Request message for ImportEntries method.

  Fields:
    gcsBucketPath: Path to a Cloud Storage bucket that contains a dump ready
      for ingestion.
    jobId: Optional. (Optional) Dataplex task job id, if specified will be
      used as part of ImportEntries LRO ID
  """
    gcsBucketPath = _messages.StringField(1)
    jobId = _messages.StringField(2)