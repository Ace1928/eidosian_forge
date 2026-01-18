from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1PlaceProductsOrderRequest(_messages.Message):
    """Request message to place an order for non-quote products.

  Fields:
    lineItemInfo: Required. Items to purchase within an order.
  """
    lineItemInfo = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1LineItemInfo', 1, repeated=True)