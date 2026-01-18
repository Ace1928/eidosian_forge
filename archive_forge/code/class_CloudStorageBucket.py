from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudStorageBucket(_messages.Message):
    """Defines the bucket by its name or a regex pattern to match buckets. Next
  ID: 3

  Fields:
    bucketName: Cloud Storage bucket name.
    bucketPrefixRegex: A regex pattern for bucket names matching the regex.
      Regex should follow the syntax specified in google/re2 on GitHub.
  """
    bucketName = _messages.StringField(1)
    bucketPrefixRegex = _messages.StringField(2)