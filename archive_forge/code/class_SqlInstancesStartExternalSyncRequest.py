from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesStartExternalSyncRequest(_messages.Message):
    """A SqlInstancesStartExternalSyncRequest object.

  Enums:
    MigrationTypeValueValuesEnum: Optional. MigrationType decides if the
      migration is a physical file based migration or logical migration.
    SyncModeValueValuesEnum: External sync mode.
    SyncParallelLevelValueValuesEnum: Optional. Parallel level for initial
      data sync. Currently only applicable for MySQL.

  Fields:
    migrationType: Optional. MigrationType decides if the migration is a
      physical file based migration or logical migration.
    mysqlSyncConfig: MySQL-specific settings for start external sync.
    skipVerification: Whether to skip the verification step (VESS).
    syncMode: External sync mode.
    syncParallelLevel: Optional. Parallel level for initial data sync.
      Currently only applicable for MySQL.
  """

    class MigrationTypeValueValuesEnum(_messages.Enum):
        """Optional. MigrationType decides if the migration is a physical file
    based migration or logical migration.

    Values:
      MIGRATION_TYPE_UNSPECIFIED: If no migration type is specified it will be
        defaulted to LOGICAL.
      LOGICAL: Logical Migrations
      PHYSICAL: Physical file based Migrations
    """
        MIGRATION_TYPE_UNSPECIFIED = 0
        LOGICAL = 1
        PHYSICAL = 2

    class SyncModeValueValuesEnum(_messages.Enum):
        """External sync mode.

    Values:
      EXTERNAL_SYNC_MODE_UNSPECIFIED: Unknown external sync mode, will be
        defaulted to ONLINE mode
      ONLINE: Online external sync will set up replication after initial data
        external sync
      OFFLINE: Offline external sync only dumps and loads a one-time snapshot
        of the primary instance's data
    """
        EXTERNAL_SYNC_MODE_UNSPECIFIED = 0
        ONLINE = 1
        OFFLINE = 2

    class SyncParallelLevelValueValuesEnum(_messages.Enum):
        """Optional. Parallel level for initial data sync. Currently only
    applicable for MySQL.

    Values:
      EXTERNAL_SYNC_PARALLEL_LEVEL_UNSPECIFIED: Unknown sync parallel level.
        Will be defaulted to OPTIMAL.
      MIN: Minimal parallel level.
      OPTIMAL: Optimal parallel level.
      MAX: Maximum parallel level.
    """
        EXTERNAL_SYNC_PARALLEL_LEVEL_UNSPECIFIED = 0
        MIN = 1
        OPTIMAL = 2
        MAX = 3
    migrationType = _messages.EnumField('MigrationTypeValueValuesEnum', 1)
    mysqlSyncConfig = _messages.MessageField('MySqlSyncConfig', 2)
    skipVerification = _messages.BooleanField(3)
    syncMode = _messages.EnumField('SyncModeValueValuesEnum', 4)
    syncParallelLevel = _messages.EnumField('SyncParallelLevelValueValuesEnum', 5)