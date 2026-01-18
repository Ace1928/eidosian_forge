from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryUsersAliasesWatchRequest(_messages.Message):
    """A DirectoryUsersAliasesWatchRequest object.

  Enums:
    EventValueValuesEnum: Event on which subscription is intended (if
      subscribing)

  Fields:
    channel: A Channel resource to be passed as the request body.
    event: Event on which subscription is intended (if subscribing)
    userKey: Email or immutable ID of the user
  """

    class EventValueValuesEnum(_messages.Enum):
        """Event on which subscription is intended (if subscribing)

    Values:
      add: Alias Created Event
      delete: Alias Deleted Event
    """
        add = 0
        delete = 1
    channel = _messages.MessageField('Channel', 1)
    event = _messages.EnumField('EventValueValuesEnum', 2)
    userKey = _messages.StringField(3, required=True)