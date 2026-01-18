from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1AuditLogRecord(_messages.Message):
    """Definition of an Audit Log Record

  Enums:
    ActionTypeValueValuesEnum: The type of action

  Fields:
    actionTime: The time when the action takes place
    actionType: The type of action
    offerId: The offer id corresponding to the log record.
    userEmail: The email of the user taking the action. This field can be
      empty for users authenticated through 3P identity provider.
    userName: The name of the user taking the action. For users authenticated
      through 3P identity provider (BYOID), the field value format is
      described in go/byoid-data-pattern:displaying-users.
  """

    class ActionTypeValueValuesEnum(_messages.Enum):
        """The type of action

    Values:
      ACTION_TYPE_UNSPECIFIED: Default value, do not use.
      ORDER_PLACED: The action of accepting an offer.
      ORDER_CANCELLED: Order Cancelation action.
      ORDER_MODIFIED: The action of modifying an order.
    """
        ACTION_TYPE_UNSPECIFIED = 0
        ORDER_PLACED = 1
        ORDER_CANCELLED = 2
        ORDER_MODIFIED = 3
    actionTime = _messages.StringField(1)
    actionType = _messages.EnumField('ActionTypeValueValuesEnum', 2)
    offerId = _messages.StringField(3)
    userEmail = _messages.StringField(4)
    userName = _messages.StringField(5)