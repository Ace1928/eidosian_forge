from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranscriptOutputConfig(_messages.Message):
    """Specifies an optional destination for the recognition results.

  Fields:
    gcsUri: Specifies a Cloud Storage URI for the recognition results. Must be
      specified in the format: `gs://bucket_name/object_name`, and the bucket
      must already exist.
  """
    gcsUri = _messages.StringField(1)