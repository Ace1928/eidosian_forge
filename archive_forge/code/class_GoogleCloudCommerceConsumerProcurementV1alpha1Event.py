from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1Event(_messages.Message):
    """Consumer Procurement Order Event

  Enums:
    EventTypeValueValuesEnum: The type of action

  Fields:
    eventTime: The time when the event takes place
    eventType: The type of action
    name: Immutable. The resource name of the order event Format:
      `billingAccounts/{billing_account}/orders/{order}/events/{event}`
    offerId: The offer id corresponding to the event.
    userEmail: The email of the user taking the action. This field can be
      empty for users authenticated through 3P identity provider.
    userName: The name of the user taking the action. For users authenticated
      through 3P identity provider (BYOID), the field value format is
      described in go/byoid-data-pattern:displaying-users.
  """

    class EventTypeValueValuesEnum(_messages.Enum):
        """The type of action

    Values:
      EVENT_TYPE_UNSPECIFIED: Default value, do not use.
      ORDER_PLACED: The action of accepting an offer.
      ORDER_CANCELLED: The action of cancelling an order.
      ORDER_MODIFIED: The action of modifying an order.
    """
        EVENT_TYPE_UNSPECIFIED = 0
        ORDER_PLACED = 1
        ORDER_CANCELLED = 2
        ORDER_MODIFIED = 3
    eventTime = _messages.StringField(1)
    eventType = _messages.EnumField('EventTypeValueValuesEnum', 2)
    name = _messages.StringField(3)
    offerId = _messages.StringField(4)
    userEmail = _messages.StringField(5)
    userName = _messages.StringField(6)