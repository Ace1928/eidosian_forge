from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UniformBucketLevelAccessValue(_messages.Message):
    """The bucket's uniform bucket-level access configuration.

      Fields:
        enabled: If set, access is controlled only by bucket-level or above
          IAM policies.
        lockedTime: The deadline for changing
          iamConfiguration.uniformBucketLevelAccess.enabled from true to false
          in RFC 3339  format.
          iamConfiguration.uniformBucketLevelAccess.enabled may be changed
          from true to false until the locked time, after which the field is
          immutable.
      """
    enabled = _messages.BooleanField(1)
    lockedTime = _message_types.DateTimeField(2)