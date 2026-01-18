from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1ListConsentsResponse(_messages.Message):
    """Response message for the list consent request.

  Fields:
    consents: Consents matching the request.
    nextPageToken: Pagination token for large results.
  """
    consents = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1Consent', 1, repeated=True)
    nextPageToken = _messages.StringField(2)