from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1ListOrdersResponse(_messages.Message):
    """Response message for ConsumerProcurementService.ListOrders.

  Fields:
    nextPageToken: The token for fetching the next page.
    orders: The list of orders in this response.
  """
    nextPageToken = _messages.StringField(1)
    orders = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1Order', 2, repeated=True)