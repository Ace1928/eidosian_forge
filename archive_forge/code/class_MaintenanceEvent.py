from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaintenanceEvent(_messages.Message):
    """A Maintenance Event is an operation that could cause temporary
  disruptions to the cluster workloads, including Google-driven or user-
  initiated cluster upgrades, user-initiated cluster configuration changes
  that require restarting nodes, etc.

  Enums:
    ScheduleValueValuesEnum: Output only. The schedule of the maintenance
      event.
    StateValueValuesEnum: Output only. The state of the maintenance event.
    TypeValueValuesEnum: Output only. The type of the maintenance event.

  Fields:
    createTime: Output only. The time when the maintenance event request was
      created.
    endTime: Output only. The time when the maintenance event ended, either
      successfully or not. If the maintenance event is split into multiple
      maintenance windows, end_time is only updated when the whole flow ends.
    operation: Output only. The operation for running the maintenance event.
      Specified in the format projects/*/locations/*/operations/*. If the
      maintenance event is split into multiple operations (e.g. due to
      maintenance windows), the latest one is recorded.
    schedule: Output only. The schedule of the maintenance event.
    startTime: Output only. The time when the maintenance event started.
    state: Output only. The state of the maintenance event.
    targetVersion: Output only. The target version of the cluster.
    type: Output only. The type of the maintenance event.
    updateTime: Output only. The time when the maintenance event message was
      updated.
    uuid: Output only. UUID of the maintenance event.
  """

    class ScheduleValueValuesEnum(_messages.Enum):
        """Output only. The schedule of the maintenance event.

    Values:
      SCHEDULE_UNSPECIFIED: Unspecified.
      IMMEDIATELY: Immediately after receiving the request.
    """
        SCHEDULE_UNSPECIFIED = 0
        IMMEDIATELY = 1

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the maintenance event.

    Values:
      STATE_UNSPECIFIED: Unspecified.
      RECONCILING: The maintenance event is ongoing. The cluster might be
        unusable.
      SUCCEEDED: The maintenance event succeeded.
      FAILED: The maintenance event failed.
      STOPPED_BEFORE_MAINTENANCE_WINDOW_ENDED: The maintenance event is
        paused. The cluster should be usable.
    """
        STATE_UNSPECIFIED = 0
        RECONCILING = 1
        SUCCEEDED = 2
        FAILED = 3
        STOPPED_BEFORE_MAINTENANCE_WINDOW_ENDED = 4

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. The type of the maintenance event.

    Values:
      TYPE_UNSPECIFIED: Unspecified.
      USER_INITIATED_UPGRADE: Upgrade initiated by users.
      GOOGLE_DRIVEN_UPGRADE: Upgrade driven by Google.
    """
        TYPE_UNSPECIFIED = 0
        USER_INITIATED_UPGRADE = 1
        GOOGLE_DRIVEN_UPGRADE = 2
    createTime = _messages.StringField(1)
    endTime = _messages.StringField(2)
    operation = _messages.StringField(3)
    schedule = _messages.EnumField('ScheduleValueValuesEnum', 4)
    startTime = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    targetVersion = _messages.StringField(7)
    type = _messages.EnumField('TypeValueValuesEnum', 8)
    updateTime = _messages.StringField(9)
    uuid = _messages.StringField(10)