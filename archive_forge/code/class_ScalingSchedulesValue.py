from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ScalingSchedulesValue(_messages.Message):
    """Scaling schedules defined for an autoscaler. Multiple schedules can be
    set on an autoscaler, and they can overlap. During overlapping periods the
    greatest min_required_replicas of all scaling schedules is applied. Up to
    128 scaling schedules are allowed.

    Messages:
      AdditionalProperty: An additional property for a ScalingSchedulesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ScalingSchedulesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ScalingSchedulesValue object.

      Fields:
        key: Name of the additional property.
        value: A AutoscalingPolicyScalingSchedule attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('AutoscalingPolicyScalingSchedule', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)