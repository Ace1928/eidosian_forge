from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1ModifyQuoteOrderRequest(_messages.Message):
    """Request message for ConsumerProcurementService.ModifyOrder.

  Enums:
    AutoRenewalBehaviorValueValuesEnum: Auto renewal behavior of the
      subscription for the update. Applied when change_type is
      [QuoteChangeType.QUOTE_CHANGE_TYPE_UPDATE].
    ChangeTypeValueValuesEnum: Required. Type of change to make.

  Fields:
    autoRenewalBehavior: Auto renewal behavior of the subscription for the
      update. Applied when change_type is
      [QuoteChangeType.QUOTE_CHANGE_TYPE_UPDATE].
    changeType: Required. Type of change to make.
    newQuoteExternalName: External name of the new quote to update to.
      Required when change_type is [QuoteChangeType.QUOTE_CHANGE_TYPE_UPDATE].
  """

    class AutoRenewalBehaviorValueValuesEnum(_messages.Enum):
        """Auto renewal behavior of the subscription for the update. Applied when
    change_type is [QuoteChangeType.QUOTE_CHANGE_TYPE_UPDATE].

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
      QUOTE_CHANGE_TYPE_UNSPECIFIED: Sentinel value. Do not use.
      QUOTE_CHANGE_TYPE_UPDATE: The change is to update an existing order for
        quote.
      QUOTE_CHANGE_TYPE_CANCEL: The change is to cancel an existing order for
        quote.
      QUOTE_CHANGE_TYPE_REVERT_CANCELLATION: The change is to revert a
        cancellation for an order for quote.
    """
        QUOTE_CHANGE_TYPE_UNSPECIFIED = 0
        QUOTE_CHANGE_TYPE_UPDATE = 1
        QUOTE_CHANGE_TYPE_CANCEL = 2
        QUOTE_CHANGE_TYPE_REVERT_CANCELLATION = 3
    autoRenewalBehavior = _messages.EnumField('AutoRenewalBehaviorValueValuesEnum', 1)
    changeType = _messages.EnumField('ChangeTypeValueValuesEnum', 2)
    newQuoteExternalName = _messages.StringField(3)