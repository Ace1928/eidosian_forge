from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1Backup(_messages.Message):
    """A Backup of a Cloud Firestore Database. The backup contains all
  documents and index configurations for the given database at a specific
  point in time.

  Enums:
    StateValueValuesEnum: Output only. The current state of the backup.

  Fields:
    database: Output only. Name of the Firestore database that the backup is
      from. Format is `projects/{project}/databases/{database}`.
    databaseUid: Output only. The system-generated UUID4 for the Firestore
      database that the backup is from.
    expireTime: Output only. The timestamp at which this backup expires.
    name: Output only. The unique resource name of the Backup. Format is
      `projects/{project}/locations/{location}/backups/{backup}`.
    snapshotTime: Output only. The backup contains an externally consistent
      copy of the database at this time.
    state: Output only. The current state of the backup.
    stats: Output only. Statistics about the backup. This data only becomes
      available after the backup is fully materialized to secondary storage.
      This field will be empty till then.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the backup.

    Values:
      STATE_UNSPECIFIED: The state is unspecified.
      CREATING: The pending backup is still being created. Operations on the
        backup will be rejected in this state.
      READY: The backup is complete and ready to use.
      NOT_AVAILABLE: The backup is not available at this moment.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        NOT_AVAILABLE = 3
    database = _messages.StringField(1)
    databaseUid = _messages.StringField(2)
    expireTime = _messages.StringField(3)
    name = _messages.StringField(4)
    snapshotTime = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    stats = _messages.MessageField('GoogleFirestoreAdminV1Stats', 7)