from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BucketErrors(_messages.Message):
    """Provides a summary of the bucket level error stats.

  Fields:
    internalErrorCount: Optional. Buckets that were not validated due to
      internal errors and will be automatically retried.
    permissionDeniedBucketIds: Optional. Subset of bucket names that have
      permission denied.
    permissionDeniedCount: Optional. Count of buckets with permission denied
      errors.
    validatedCount: Optional. Count of successfully validated buckets.
  """
    internalErrorCount = _messages.IntegerField(1)
    permissionDeniedBucketIds = _messages.StringField(2, repeated=True)
    permissionDeniedCount = _messages.IntegerField(3)
    validatedCount = _messages.IntegerField(4)