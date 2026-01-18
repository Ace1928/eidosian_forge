from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementBillingAccountsOrdersOrderAttributionsListRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementBillingAccountsOrdersOrderAttributions
  ListRequest object.

  Fields:
    pageSize: The maximum number of entries returned per call.
    pageToken: The token for fetching the next page of entries.
    parent: Required. The parent Order to query for OrderAttributions. This
      field is of the form `billingAccounts/{billing-account-
      id}/orders/{order-id}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)