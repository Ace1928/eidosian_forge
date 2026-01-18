from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1Order(_messages.Message):
    """Represents a purchase made by a customer on Cloud Marketplace. Creating
  an order makes sure that both the Google backend systems as well as external
  service provider's systems (if needed) allow use of purchased products and
  ensures the appropriate billing events occur. An Order can be made against
  one Product with multiple add-ons (optional) or one Quote which might
  reference multiple products. Customers typically choose a price plan for
  each Product purchased when they create an order and can change their plan
  later, if the product allows.

  Enums:
    OrderStateValueValuesEnum: Output only. The state of the order.

  Fields:
    account: The resource name of the account that this order is based on.
      Required if the creation of any products in the order requires an
      account to be present.
    cancelledLineItems: Output only. Line items that were cancelled.
    createTime: Output only. The creation timestamp.
    displayName: Required. The user-specified name of the order.
    etag: The weak etag of the order.
    lineItems: Output only. The items being purchased.
    name: Output only. The resource name of the order. Has the form
      `billingAccounts/{billing_account}/orders/{order}`.
    orderState: Output only. The state of the order.
    provider: Provider of the products being purchased. Provider has the
      format of `providers/{provider}`.
    stateReason: Output only. An explanation for the order's state. Mainly
      used in the case of `OrderState.ORDER_STATE_CANCELLED` states to explain
      why the order is cancelled.
    updateTime: Output only. The last update timestamp.
  """

    class OrderStateValueValuesEnum(_messages.Enum):
        """Output only. The state of the order.

    Values:
      ORDER_STATE_UNSPECIFIED: Sentinel value. Do not use.
      ORDER_STATE_ACTIVE: The order is active.
      ORDER_STATE_CANCELLED: The order is cancelled.
      ORDER_STATE_PENDING_CANCELLATION: The order is being cancelled either by
        the user or by the system. The order stays in this state, if any
        product in the order allows use of the underlying resource until the
        end of the current billing cycle. Once the billing cycle completes,
        the resource will transition to OrderState.ORDER_STATE_CANCELLED
        state.
    """
        ORDER_STATE_UNSPECIFIED = 0
        ORDER_STATE_ACTIVE = 1
        ORDER_STATE_CANCELLED = 2
        ORDER_STATE_PENDING_CANCELLATION = 3
    account = _messages.StringField(1)
    cancelledLineItems = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1LineItem', 2, repeated=True)
    createTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    etag = _messages.StringField(5)
    lineItems = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1LineItem', 6, repeated=True)
    name = _messages.StringField(7)
    orderState = _messages.EnumField('OrderStateValueValuesEnum', 8)
    provider = _messages.StringField(9)
    stateReason = _messages.StringField(10)
    updateTime = _messages.StringField(11)