from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MoveBucketResponse(_messages.Message):
    """The response from MoveBucket.

  Fields:
    bucket: The resulting bucket from the move action.
  """
    bucket = _messages.MessageField('LogBucket', 1)