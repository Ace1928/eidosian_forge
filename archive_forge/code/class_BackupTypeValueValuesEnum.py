from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupTypeValueValuesEnum(_messages.Enum):
    """BackupTypeValueValuesEnum enum type.

    Values:
      BACKUP_TYPE_UNSPECIFIED: <no description>
      SCHEDULED: <no description>
      ON_DEMAND: <no description>
    """
    BACKUP_TYPE_UNSPECIFIED = 0
    SCHEDULED = 1
    ON_DEMAND = 2