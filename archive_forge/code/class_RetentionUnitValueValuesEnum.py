from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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