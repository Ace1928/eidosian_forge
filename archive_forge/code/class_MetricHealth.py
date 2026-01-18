from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricHealth(_messages.Message):
    """Metric health, such as used_memory_ratio, redis_server_cpu_usage

  Enums:
    StateValueValuesEnum: Health state, such as unhealthy/warning/healthy

  Fields:
    metric: Name of this metric
    reason: Reason if the status is not healthy
    state: Health state, such as unhealthy/warning/healthy
    suggestion: Actional suggestion if the status is not healthy
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
    metric = _messages.StringField(1)
    reason = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    suggestion = _messages.StringField(4)