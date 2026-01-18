from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BackupRetentionSettings(_messages.Message):
    """We currently only support backup retention by specifying the number of
  backups we will retain.

  Enums:
    RetentionUnitValueValuesEnum: The unit that 'retained_backups' represents.

  Fields:
    retainedBackups: Depending on the value of retention_unit, this is used to
      determine if a backup needs to be deleted. If retention_unit is 'COUNT',
      we will retain this many backups.
    retentionUnit: The unit that 'retained_backups' represents.
  """

    class RetentionUnitValueValuesEnum(_messages.Enum):
        """The unit that 'retained_backups' represents.

    Values:
      RETENTION_UNIT_UNSPECIFIED: Backup retention unit is unspecified, will
        be treated as COUNT.
      COUNT: Retention will be by count, eg. "retain the most recent 7
        backups".
    """
        RETENTION_UNIT_UNSPECIFIED = 0
        COUNT = 1
    retainedBackups = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    retentionUnit = _messages.EnumField('RetentionUnitValueValuesEnum', 2)