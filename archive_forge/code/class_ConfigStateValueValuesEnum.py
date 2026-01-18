from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigStateValueValuesEnum(_messages.Enum):
    """The backup configuration state.

    Values:
      BACKUP_CONFIG_STATE_UNSPECIFIED: The possible states of backup
        configuration. Status not set.
      ACTIVE: The data source is actively protected (i.e. there is a
        BackupPlanAssociation or Appliance SLA pointing to it)
      PASSIVE: The data source is no longer protected (but may have backups
        under it)
    """
    BACKUP_CONFIG_STATE_UNSPECIFIED = 0
    ACTIVE = 1
    PASSIVE = 2