from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1ModifyProductsOrderRequestModification(_messages.Message):
    """Modifications to make on the order.

  Enums:
    AutoRenewalBehaviorValueValuesEnum: Auto renewal behavior of the
      subscription for the update. Applied when change_type is
      [LineItemChangeType.LINE_ITEM_CHANGE_TYPE_UPDATE].
    ChangeTypeValueValuesEnum: Required. Type of change to make.

  Fields:
    autoRenewalBehavior: Auto renewal behavior of the subscription for the
      update. Applied when change_type is
      [LineItemChangeType.LINE_ITEM_CHANGE_TYPE_UPDATE].
    changeType: Required. Type of change to make.
    lineItemId: ID of the existing line item to make change to. Required when
      change type is [LineItemChangeType.LINE_ITEM_CHANGE_TYPE_UPDATE] or
      [LineItemChangeType.LINE_ITEM_CHANGE_TYPE_CANCEL].
    newLineItemInfo: The line item to update to. Required when change_type is
      [LineItemChangeType.LINE_ITEM_CHANGE_TYPE_CREATE] or
      [LineItemChangeType.LINE_ITEM_CHANGE_TYPE_UPDATE].
  """

    class AutoRenewalBehaviorValueValuesEnum(_messages.Enum):
        """Auto renewal behavior of the subscription for the update. Applied when
    change_type is [LineItemChangeType.LINE_ITEM_CHANGE_TYPE_UPDATE].

    Values:
      AUTO_RENEWAL_BEHAVIOR_UNSPECIFIED: If unspecified, the auto renewal
        behavior will follow the default config.
      AUTO_RENEWAL_BEHAVIOR_ENABLE: Auto Renewal will be enabled on
        subscription.
      AUTO_RENEWAL_BEHAVIOR_DISABLE: Auto Renewal will be disabled on
        subscription.
    """
        AUTO_RENEWAL_BEHAVIOR_UNSPECIFIED = 0
        AUTO_RENEWAL_BEHAVIOR_ENABLE = 1
        AUTO_RENEWAL_BEHAVIOR_DISABLE = 2

    class ChangeTypeValueValuesEnum(_messages.Enum):
        """Required. Type of change to make.

    Values:
      LINE_ITEM_CHANGE_TYPE_UNSPECIFIED: Sentinel value. Do not use.
      LINE_ITEM_CHANGE_TYPE_CREATE: The change is to create a new line item.
      LINE_ITEM_CHANGE_TYPE_UPDATE: The change is to update an existing line
        item.
      LINE_ITEM_CHANGE_TYPE_CANCEL: The change is to cancel an existing line
        item.
      LINE_ITEM_CHANGE_TYPE_REVERT_CANCELLATION: The change is to revert a
        cancellation.
    """
        LINE_ITEM_CHANGE_TYPE_UNSPECIFIED = 0
        LINE_ITEM_CHANGE_TYPE_CREATE = 1
        LINE_ITEM_CHANGE_TYPE_UPDATE = 2
        LINE_ITEM_CHANGE_TYPE_CANCEL = 3
        LINE_ITEM_CHANGE_TYPE_REVERT_CANCELLATION = 4
    autoRenewalBehavior = _messages.EnumField('AutoRenewalBehaviorValueValuesEnum', 1)
    changeType = _messages.EnumField('ChangeTypeValueValuesEnum', 2)
    lineItemId = _messages.StringField(3)
    newLineItemInfo = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1LineItemInfo', 4)