from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeClusterRequest(_messages.Message):
    """Upgrades a cluster.

  Enums:
    ScheduleValueValuesEnum: The schedule for the upgrade.

  Fields:
    requestId: A unique identifier for this request. Restricted to 36 ASCII
      characters. A random UUID is recommended. This request is only
      idempotent if `request_id` is provided.
    schedule: The schedule for the upgrade.
    targetVersion: Required. The version the cluster is going to be upgraded
      to.
  """

    class ScheduleValueValuesEnum(_messages.Enum):
        """The schedule for the upgrade.

    Values:
      SCHEDULE_UNSPECIFIED: Unspecified. The default is to upgrade the cluster
        immediately which is the only option today.
      IMMEDIATELY: The cluster is going to be upgraded immediately after
        receiving the request.
    """
        SCHEDULE_UNSPECIFIED = 0
        IMMEDIATELY = 1
    requestId = _messages.StringField(1)
    schedule = _messages.EnumField('ScheduleValueValuesEnum', 2)
    targetVersion = _messages.StringField(3)