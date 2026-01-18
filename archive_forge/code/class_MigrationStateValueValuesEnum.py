from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MigrationStateValueValuesEnum(_messages.Enum):
    """Output only. The migration state of the alias.

    Values:
      MIGRATION_STATE_UNSPECIFIED: Default migration state. This value should
        never be used.
      PENDING: Indicates migration is yet to be performed.
      COMPLETED: Indicates migration is successfully completed.
      IN_PROGRESS: Indicates migration in progress.
      NOT_REQUIRED: This value indicates there was no migration state for the
        alias defined in the system. This would be the case for the aliases
        that are new i.e., do not have resource at the time of cutover.
    """
    MIGRATION_STATE_UNSPECIFIED = 0
    PENDING = 1
    COMPLETED = 2
    IN_PROGRESS = 3
    NOT_REQUIRED = 4