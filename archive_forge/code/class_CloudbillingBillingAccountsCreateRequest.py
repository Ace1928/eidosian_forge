from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudbillingBillingAccountsCreateRequest(_messages.Message):
    """A CloudbillingBillingAccountsCreateRequest object.

  Fields:
    billingAccount: A BillingAccount resource to be passed as the request
      body.
    parent: Optional. The parent to create a billing account from. Format: -
      `billingAccounts/{billing_account_id}`, for example,
      `billingAccounts/012345-567890-ABCDEF`
  """
    billingAccount = _messages.MessageField('BillingAccount', 1)
    parent = _messages.StringField(2)