from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2StoredInfoTypeVersion(_messages.Message):
    """Version of a StoredInfoType, including the configuration used to build
  it, create timestamp, and current state.

  Enums:
    StateValueValuesEnum: Stored info type version state. Read-only, updated
      by the system during dictionary creation.

  Fields:
    config: StoredInfoType configuration.
    createTime: Create timestamp of the version. Read-only, determined by the
      system when the version is created.
    errors: Errors that occurred when creating this storedInfoType version, or
      anomalies detected in the storedInfoType data that render it unusable.
      Only the five most recent errors will be displayed, with the most recent
      error appearing first. For example, some of the data for stored custom
      dictionaries is put in the user's Cloud Storage bucket, and if this data
      is modified or deleted by the user or another system, the dictionary
      becomes invalid. If any errors occur, fix the problem indicated by the
      error message and use the UpdateStoredInfoType API method to create
      another version of the storedInfoType to continue using it, reusing the
      same `config` if it was not the source of the error.
    state: Stored info type version state. Read-only, updated by the system
      during dictionary creation.
    stats: Statistics about this storedInfoType version.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Stored info type version state. Read-only, updated by the system
    during dictionary creation.

    Values:
      STORED_INFO_TYPE_STATE_UNSPECIFIED: Unused
      PENDING: StoredInfoType version is being created.
      READY: StoredInfoType version is ready for use.
      FAILED: StoredInfoType creation failed. All relevant error messages are
        returned in the `StoredInfoTypeVersion` message.
      INVALID: StoredInfoType is no longer valid because artifacts stored in
        user-controlled storage were modified. To fix an invalid
        StoredInfoType, use the `UpdateStoredInfoType` method to create a new
        version.
    """
        STORED_INFO_TYPE_STATE_UNSPECIFIED = 0
        PENDING = 1
        READY = 2
        FAILED = 3
        INVALID = 4
    config = _messages.MessageField('GooglePrivacyDlpV2StoredInfoTypeConfig', 1)
    createTime = _messages.StringField(2)
    errors = _messages.MessageField('GooglePrivacyDlpV2Error', 3, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    stats = _messages.MessageField('GooglePrivacyDlpV2StoredInfoTypeStats', 5)