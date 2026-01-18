from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1GcsFileSpec(_messages.Message):
    """Specification of a single file in Cloud Storage.

  Fields:
    filePath: Required. Full file path. Example: `gs://bucket_name/a/b.txt`.
    gcsTimestamps: Output only. Creation, modification, and expiration
      timestamps of a Cloud Storage file.
    sizeBytes: Output only. File size in bytes.
  """
    filePath = _messages.StringField(1)
    gcsTimestamps = _messages.MessageField('GoogleCloudDatacatalogV1SystemTimestamps', 2)
    sizeBytes = _messages.IntegerField(3)