from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBucketInfo(_messages.Message):
    """For display only. Metadata associated with Storage Bucket.

  Fields:
    bucket: Cloud Storage Bucket name.
  """
    bucket = _messages.StringField(1)