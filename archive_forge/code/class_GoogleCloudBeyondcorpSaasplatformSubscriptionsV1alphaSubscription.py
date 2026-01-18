from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpSaasplatformSubscriptionsV1alphaSubscription(_messages.Message):
    """A BeyondCorp Subscription resource represents BeyondCorp Enterprise
  Subscription. BeyondCorp Enterprise Subscription enables BeyondCorp
  Enterprise permium features for an organization.

  Enums:
    SkuValueValuesEnum: Required. SKU of subscription.
    StateValueValuesEnum: Output only. The current state of the subscription.
    TypeValueValuesEnum: Required. Type of subscription.

  Fields:
    autoRenewEnabled: Output only. Represents that, if subscription will renew
      or end when the term ends.
    createTime: Output only. Create time of the subscription.
    endTime: Output only. End time of the subscription.
    name: Required. Unique resource name of the Subscription. The name is
      ignored when creating a subscription.
    seatCount: Optional. Number of seats in the subscription.
    sku: Required. SKU of subscription.
    startTime: Output only. Start time of the subscription.
    state: Output only. The current state of the subscription.
    type: Required. Type of subscription.
  """

    class SkuValueValuesEnum(_messages.Enum):
        """Required. SKU of subscription.

    Values:
      SKU_UNSPECIFIED: Default value. This value is unused.
      BCE_STANDARD_SKU: Represents BeyondCorp Standard SKU.
    """
        SKU_UNSPECIFIED = 0
        BCE_STANDARD_SKU = 1

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the subscription.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      ACTIVE: Represents an active subscription.
      INACTIVE: Represents an upcomming subscription.
      COMPLETED: Represents a completed subscription.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        INACTIVE = 2
        COMPLETED = 3

    class TypeValueValuesEnum(_messages.Enum):
        """Required. Type of subscription.

    Values:
      TYPE_UNSPECIFIED: Default value. This value is unused.
      TRIAL: Represents a trial subscription.
      PAID: Represents a paid subscription.
      ALLOWLIST: Reresents an allowlisted subscription.
    """
        TYPE_UNSPECIFIED = 0
        TRIAL = 1
        PAID = 2
        ALLOWLIST = 3
    autoRenewEnabled = _messages.BooleanField(1)
    createTime = _messages.StringField(2)
    endTime = _messages.StringField(3)
    name = _messages.StringField(4)
    seatCount = _messages.IntegerField(5)
    sku = _messages.EnumField('SkuValueValuesEnum', 6)
    startTime = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    type = _messages.EnumField('TypeValueValuesEnum', 9)