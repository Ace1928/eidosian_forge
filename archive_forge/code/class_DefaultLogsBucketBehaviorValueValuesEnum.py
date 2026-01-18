from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DefaultLogsBucketBehaviorValueValuesEnum(_messages.Enum):
    """Optional. Option to specify how default logs buckets are setup.

    Values:
      DEFAULT_LOGS_BUCKET_BEHAVIOR_UNSPECIFIED: Unspecified.
      REGIONAL_USER_OWNED_BUCKET: Bucket is located in user-owned project in
        the same region as the build. The builder service account must have
        access to create and write to Cloud Storage buckets in the build
        project.
    """
    DEFAULT_LOGS_BUCKET_BEHAVIOR_UNSPECIFIED = 0
    REGIONAL_USER_OWNED_BUCKET = 1