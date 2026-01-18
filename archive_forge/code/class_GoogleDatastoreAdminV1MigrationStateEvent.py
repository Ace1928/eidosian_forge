from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1MigrationStateEvent(_messages.Message):
    """An event signifying a change in state of a [migration from Cloud
  Datastore to Cloud Firestore in Datastore
  mode](https://cloud.google.com/datastore/docs/upgrade-to-firestore).

  Enums:
    StateValueValuesEnum: The new state of the migration.

  Fields:
    state: The new state of the migration.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The new state of the migration.

    Values:
      MIGRATION_STATE_UNSPECIFIED: Unspecified.
      RUNNING: The migration is running.
      PAUSED: The migration is paused.
      COMPLETE: The migration is complete.
    """
        MIGRATION_STATE_UNSPECIFIED = 0
        RUNNING = 1
        PAUSED = 2
        COMPLETE = 3
    state = _messages.EnumField('StateValueValuesEnum', 1)