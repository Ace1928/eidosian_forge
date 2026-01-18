from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1ListOrderAttributionsResponse(_messages.Message):
    """Response message for ConsumerProcurementService.ListOrderAttributions.

  Fields:
    nextPageToken: The token for fetching the next page of entries.
    orderAttributions: The OrderAttributions from this response
  """
    nextPageToken = _messages.StringField(1)
    orderAttributions = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1OrderAttribution', 2, repeated=True)