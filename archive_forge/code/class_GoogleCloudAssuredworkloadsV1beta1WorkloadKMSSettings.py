from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1beta1WorkloadKMSSettings(_messages.Message):
    """Settings specific to the Key Management Service.

  Fields:
    nextRotationTime: Required. Input only. Immutable. The time at which the
      Key Management Service will automatically create a new version of the
      crypto key and mark it as the primary.
    rotationPeriod: Required. Input only. Immutable. [next_rotation_time] will
      be advanced by this period when the Key Management Service automatically
      rotates a key. Must be at least 24 hours and at most 876,000 hours.
  """
    nextRotationTime = _messages.StringField(1)
    rotationPeriod = _messages.StringField(2)