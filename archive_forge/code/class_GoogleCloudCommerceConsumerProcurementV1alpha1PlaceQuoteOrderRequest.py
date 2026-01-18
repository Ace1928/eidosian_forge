from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1PlaceQuoteOrderRequest(_messages.Message):
    """Request message to place an order for a quote.

  Fields:
    quoteExternalName: Required. External name of the quote to purchase.
  """
    quoteExternalName = _messages.StringField(1)