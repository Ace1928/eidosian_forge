from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDatabasecenterPartnerapiV1mainRetentionSettings(_messages.Message):
    """A StorageDatabasecenterPartnerapiV1mainRetentionSettings object.

  Enums:
    RetentionUnitValueValuesEnum: The unit that 'retained_backups' represents.

  Fields:
    quantityBasedRetention: A integer attribute.
    retentionUnit: The unit that 'retained_backups' represents.
    timeBasedRetention: A string attribute.
  """

    class RetentionUnitValueValuesEnum(_messages.Enum):
        """The unit that 'retained_backups' represents.

    Values:
      RETENTION_UNIT_UNSPECIFIED: Backup retention unit is unspecified, will
        be treated as COUNT.
      COUNT: Retention will be by count, eg. "retain the most recent 7
        backups".
      TIME: Retention will be by Time, eg. "retain the last 7 days backups".
      RETENTION_UNIT_OTHER: For rest of the other category
    """
        RETENTION_UNIT_UNSPECIFIED = 0
        COUNT = 1
        TIME = 2
        RETENTION_UNIT_OTHER = 3
    quantityBasedRetention = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    retentionUnit = _messages.EnumField('RetentionUnitValueValuesEnum', 2)
    timeBasedRetention = _messages.StringField(3)