from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudbillingBillingAccountsMoveRequest(_messages.Message):
    """A CloudbillingBillingAccountsMoveRequest object.

  Fields:
    moveBillingAccountRequest: A MoveBillingAccountRequest resource to be
      passed as the request body.
    name: Required. The resource name of the billing account to move. Must be
      of the form `billingAccounts/{billing_account_id}`. The specified
      billing account cannot be a subaccount, since a subaccount always
      belongs to the same organization as its parent account.
  """
    moveBillingAccountRequest = _messages.MessageField('MoveBillingAccountRequest', 1)
    name = _messages.StringField(2, required=True)