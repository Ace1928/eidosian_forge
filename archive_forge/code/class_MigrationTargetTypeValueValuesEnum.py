from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MigrationTargetTypeValueValuesEnum(_messages.Enum):
    """Immutable. The target type of this group.

    Values:
      MIGRATION_TARGET_TYPE_UNSPECIFIED: Group type is not specified. This
        defaults to Compute Engine targets.
      MIGRATION_TARGET_TYPE_GCE: All MigratingVMs in the group must have
        Compute Engine targets.
      MIGRATION_TARGET_TYPE_DISKS: All MigratingVMs in the group must have
        Compute Engine Disks targets.
    """
    MIGRATION_TARGET_TYPE_UNSPECIFIED = 0
    MIGRATION_TARGET_TYPE_GCE = 1
    MIGRATION_TARGET_TYPE_DISKS = 2