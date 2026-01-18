from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1ModifyOrderRequest(_messages.Message):
    """Request message for ConsumerProcurementService.ModifyOrder. Next Id: 10

  Fields:
    displayName: Optional. Updated display name of the order, leave as empty
      if you do not want to update current display name.
    etag: The weak etag, which can be optionally populated, of the order that
      this modify request is based on. Validation checking will only happen if
      the invoker supplies this field.
    modifications: Optional. Modifications for an existing Order created by an
      Offer. Required when Offer based Order is being modified, except for
      when going from an offer to a public plan.
    modifyProductsOrderRequest: Required. Modifies an existing non-quote
      order. Should only be used for offer-based orders when going from an
      offer to a public plan.
    modifyQuoteOrderRequest: Required. Modifies an existing order for quote.
  """
    displayName = _messages.StringField(1)
    etag = _messages.StringField(2)
    modifications = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1ModifyOrderRequestModification', 3, repeated=True)
    modifyProductsOrderRequest = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1ModifyProductsOrderRequest', 4)
    modifyQuoteOrderRequest = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1ModifyQuoteOrderRequest', 5)