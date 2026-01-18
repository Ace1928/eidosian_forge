from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1ListAccountsResponse(_messages.Message):
    """Response message for ConsumerProcurementService.ListAccounts.

  Fields:
    accounts: The list of accounts in this response.
    nextPageToken: The token for fetching the next page.
  """
    accounts = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1Account', 1, repeated=True)
    nextPageToken = _messages.StringField(2)