from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScalingScheduleStatus(_messages.Message):
    """A ScalingScheduleStatus object.

  Enums:
    StateValueValuesEnum: [Output Only] The current state of a scaling
      schedule.

  Fields:
    lastStartTime: [Output Only] The last time the scaling schedule became
      active. Note: this is a timestamp when a schedule actually became
      active, not when it was planned to do so. The timestamp is in RFC3339
      text format.
    nextStartTime: [Output Only] The next time the scaling schedule is to
      become active. Note: this is a timestamp when a schedule is planned to
      run, but the actual time might be slightly different. The timestamp is
      in RFC3339 text format.
    state: [Output Only] The current state of a scaling schedule.
  """

    class StateValueValuesEnum(_messages.Enum):
        """[Output Only] The current state of a scaling schedule.

    Values:
      ACTIVE: The current autoscaling recommendation is influenced by this
        scaling schedule.
      DISABLED: This scaling schedule has been disabled by the user.
      OBSOLETE: This scaling schedule will never become active again.
      READY: The current autoscaling recommendation is not influenced by this
        scaling schedule.
    """
        ACTIVE = 0
        DISABLED = 1
        OBSOLETE = 2
        READY = 3
    lastStartTime = _messages.StringField(1)
    nextStartTime = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)