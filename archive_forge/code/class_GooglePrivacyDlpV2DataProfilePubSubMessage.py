from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DataProfilePubSubMessage(_messages.Message):
    """Pub/Sub topic message for a DataProfileAction.PubSubNotification event.
  To receive a message of protocol buffer schema type, convert the message
  data to an object of this proto class.

  Enums:
    EventValueValuesEnum: The event that caused the Pub/Sub message to be
      sent.

  Fields:
    event: The event that caused the Pub/Sub message to be sent.
    profile: If `DetailLevel` is `TABLE_PROFILE` this will be fully populated.
      Otherwise, if `DetailLevel` is `RESOURCE_NAME`, then only `name` and
      `full_resource` will be populated.
  """

    class EventValueValuesEnum(_messages.Enum):
        """The event that caused the Pub/Sub message to be sent.

    Values:
      EVENT_TYPE_UNSPECIFIED: Unused.
      NEW_PROFILE: New profile (not a re-profile).
      CHANGED_PROFILE: Changed one of the following profile metrics: * Table
        data risk score * Table sensitivity score * Table resource visibility
        * Table encryption type * Table predicted infoTypes * Table other
        infoTypes
      SCORE_INCREASED: Table data risk score or sensitivity score increased.
      ERROR_CHANGED: A user (non-internal) error occurred.
    """
        EVENT_TYPE_UNSPECIFIED = 0
        NEW_PROFILE = 1
        CHANGED_PROFILE = 2
        SCORE_INCREASED = 3
        ERROR_CHANGED = 4
    event = _messages.EnumField('EventValueValuesEnum', 1)
    profile = _messages.MessageField('GooglePrivacyDlpV2TableDataProfile', 2)