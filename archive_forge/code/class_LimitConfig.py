from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LimitConfig(_messages.Message):
    """Represents the autoscaling limit configuration of a metastore service.

  Fields:
    maxScalingFactor: Optional. The highest scaling factor that the service
      should be autoscaled to.
    minScalingFactor: Optional. The lowest scaling factor that the service
      should be autoscaled to.
  """
    maxScalingFactor = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    minScalingFactor = _messages.FloatField(2, variant=_messages.Variant.FLOAT)