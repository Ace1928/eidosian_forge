from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CategoryHealth(_messages.Message):
    """Category health, such as CPU/memory

  Enums:
    StateValueValuesEnum: Health state, such as unhealthy/warning/healthy

  Fields:
    category: Name of this category
    metrics: Associated metrics
    state: Health state, such as unhealthy/warning/healthy
  """

    class StateValueValuesEnum(_messages.Enum):
        """Health state, such as unhealthy/warning/healthy

    Values:
      HEALTH_STATE_UNSPECIFIED: Invalid
      UNKNOWN: Unknown. May indicate exceptions.
      HEALTHY: Healthy
      WARNING: Warning
      UNHEALTHY: Unhealthy
    """
        HEALTH_STATE_UNSPECIFIED = 0
        UNKNOWN = 1
        HEALTHY = 2
        WARNING = 3
        UNHEALTHY = 4
    category = _messages.StringField(1)
    metrics = _messages.MessageField('MetricHealth', 2, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 3)