from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class MigrationTypeValueValuesEnum(_messages.Enum):
    """Optional. MigrationType field decides if the migration is a physical
    file based migration or logical migration

    Values:
      MIGRATION_TYPE_UNSPECIFIED: If no migration type is specified it will be
        defaulted to LOGICAL.
      LOGICAL: Logical Migrations
      PHYSICAL: Physical file based Migrations
    """
    MIGRATION_TYPE_UNSPECIFIED = 0
    LOGICAL = 1
    PHYSICAL = 2