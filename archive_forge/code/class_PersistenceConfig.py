from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PersistenceConfig(_messages.Message):
    """Configuration of the persistence functionality.

  Enums:
    PersistenceModeValueValuesEnum: Optional. Controls whether Persistence
      features are enabled. If not provided, the existing value will be used.
    RdbSnapshotPeriodValueValuesEnum: Optional. Period between RDB snapshots.
      Snapshots will be attempted every period starting from the provided
      snapshot start time. For example, a start time of 01/01/2033 06:45 and
      SIX_HOURS snapshot period will do nothing until 01/01/2033, and then
      trigger snapshots every day at 06:45, 12:45, 18:45, and 00:45 the next
      day, and so on. If not provided, TWENTY_FOUR_HOURS will be used as
      default.

  Fields:
    persistenceMode: Optional. Controls whether Persistence features are
      enabled. If not provided, the existing value will be used.
    rdbNextSnapshotTime: Output only. The next time that a snapshot attempt is
      scheduled to occur.
    rdbSnapshotPeriod: Optional. Period between RDB snapshots. Snapshots will
      be attempted every period starting from the provided snapshot start
      time. For example, a start time of 01/01/2033 06:45 and SIX_HOURS
      snapshot period will do nothing until 01/01/2033, and then trigger
      snapshots every day at 06:45, 12:45, 18:45, and 00:45 the next day, and
      so on. If not provided, TWENTY_FOUR_HOURS will be used as default.
    rdbSnapshotStartTime: Optional. Date and time that the first snapshot
      was/will be attempted, and to which future snapshots will be aligned. If
      not provided, the current time will be used.
  """

    class PersistenceModeValueValuesEnum(_messages.Enum):
        """Optional. Controls whether Persistence features are enabled. If not
    provided, the existing value will be used.

    Values:
      PERSISTENCE_MODE_UNSPECIFIED: Not set.
      DISABLED: Persistence is disabled for the instance, and any existing
        snapshots are deleted.
      RDB: RDB based Persistence is enabled.
    """
        PERSISTENCE_MODE_UNSPECIFIED = 0
        DISABLED = 1
        RDB = 2

    class RdbSnapshotPeriodValueValuesEnum(_messages.Enum):
        """Optional. Period between RDB snapshots. Snapshots will be attempted
    every period starting from the provided snapshot start time. For example,
    a start time of 01/01/2033 06:45 and SIX_HOURS snapshot period will do
    nothing until 01/01/2033, and then trigger snapshots every day at 06:45,
    12:45, 18:45, and 00:45 the next day, and so on. If not provided,
    TWENTY_FOUR_HOURS will be used as default.

    Values:
      SNAPSHOT_PERIOD_UNSPECIFIED: Not set.
      ONE_HOUR: Snapshot every 1 hour.
      SIX_HOURS: Snapshot every 6 hours.
      TWELVE_HOURS: Snapshot every 12 hours.
      TWENTY_FOUR_HOURS: Snapshot every 24 hours.
    """
        SNAPSHOT_PERIOD_UNSPECIFIED = 0
        ONE_HOUR = 1
        SIX_HOURS = 2
        TWELVE_HOURS = 3
        TWENTY_FOUR_HOURS = 4
    persistenceMode = _messages.EnumField('PersistenceModeValueValuesEnum', 1)
    rdbNextSnapshotTime = _messages.StringField(2)
    rdbSnapshotPeriod = _messages.EnumField('RdbSnapshotPeriodValueValuesEnum', 3)
    rdbSnapshotStartTime = _messages.StringField(4)