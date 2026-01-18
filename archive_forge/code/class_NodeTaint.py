from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeTaint(_messages.Message):
    """Kubernetes taint is composed of three fields: key, value, and effect.
  Effect can only be one of three types: NoSchedule, PreferNoSchedule or
  NoExecute. See
  [here](https://kubernetes.io/docs/concepts/configuration/taint-and-
  toleration) for more information, including usage and the valid values.

  Enums:
    EffectValueValuesEnum: Effect for taint.

  Fields:
    effect: Effect for taint.
    key: Key for taint.
    value: Value for taint.
  """

    class EffectValueValuesEnum(_messages.Enum):
        """Effect for taint.

    Values:
      EFFECT_UNSPECIFIED: Not set
      NO_SCHEDULE: NoSchedule
      PREFER_NO_SCHEDULE: PreferNoSchedule
      NO_EXECUTE: NoExecute
    """
        EFFECT_UNSPECIFIED = 0
        NO_SCHEDULE = 1
        PREFER_NO_SCHEDULE = 2
        NO_EXECUTE = 3
    effect = _messages.EnumField('EffectValueValuesEnum', 1)
    key = _messages.StringField(2)
    value = _messages.StringField(3)