from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuleConfigInfo(_messages.Message):
    """Message for rules config info.

  Enums:
    StateValueValuesEnum: Output only. The config state for rule.

  Fields:
    dataSource: Output only. Output Only. Resource name of data source which
      will be used as storage location for backups taken by specified rule.
      Format : projects/{project}/locations/{location}/backupVaults/{backupvau
      lt}/dataSources/{datasource}
    lastSuccessfulBackupConsistencyTime: Output only. The point in time when
      the last successful backup was captured from the source.
    ruleId: Output only. Output Only. Backup Rule id fetched from backup plan.
    state: Output only. The config state for rule.
    stateDetails: Output only. Output Only. Additional details for current
      config state.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The config state for rule.

    Values:
      CONFIG_STATE_UNSPECIFIED: State not set.
      FIRST_BACKUP_PENDING: The first backup is pending.
      PERMISSIONS_MISSING: Required permissions are missing to perform backup
        operation.
      LAST_BACKUP_SUCCEEDED: The last backup operation succeeded.
      LAST_BACKUP_FAILED: The last backup operation failed.
    """
        CONFIG_STATE_UNSPECIFIED = 0
        FIRST_BACKUP_PENDING = 1
        PERMISSIONS_MISSING = 2
        LAST_BACKUP_SUCCEEDED = 3
        LAST_BACKUP_FAILED = 4
    dataSource = _messages.StringField(1)
    lastSuccessfulBackupConsistencyTime = _messages.StringField(2)
    ruleId = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    stateDetails = _messages.StringField(5)