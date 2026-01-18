from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventSubscriptionStatus(_messages.Message):
    """EventSubscription Status denotes the status of the EventSubscription
  resource.

  Enums:
    StateValueValuesEnum: Output only. State of Event Subscription resource.

  Fields:
    description: Output only. Description of the state.
    state: Output only. State of Event Subscription resource.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of Event Subscription resource.

    Values:
      STATE_UNSPECIFIED: Default state.
      CREATING: EventSubscription creation is in progress.
      UPDATING: EventSubscription is in Updating status.
      ACTIVE: EventSubscription is in Active state and is ready to receive
        events.
      SUSPENDED: EventSubscription is currently suspended.
      ERROR: EventSubscription is in Error state.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        UPDATING = 2
        ACTIVE = 3
        SUSPENDED = 4
        ERROR = 5
    description = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)