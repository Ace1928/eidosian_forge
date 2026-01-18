from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudStorageBuckets(_messages.Message):
    """Collection of Cloud Storage buckets. Next ID: 2

  Fields:
    cloudStorageBuckets: A CloudStorageBucket attribute.
  """
    cloudStorageBuckets = _messages.MessageField('CloudStorageBucket', 1, repeated=True)