from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudbillingBillingAccountsGetRequest(_messages.Message):
    """A CloudbillingBillingAccountsGetRequest object.

  Fields:
    name: Required. The resource name of the billing account to retrieve. For
      example, `billingAccounts/012345-567890-ABCDEF`.
  """
    name = _messages.StringField(1, required=True)