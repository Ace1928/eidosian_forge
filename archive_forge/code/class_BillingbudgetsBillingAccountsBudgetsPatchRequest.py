from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BillingbudgetsBillingAccountsBudgetsPatchRequest(_messages.Message):
    """A BillingbudgetsBillingAccountsBudgetsPatchRequest object.

  Fields:
    googleCloudBillingBudgetsV1Budget: A GoogleCloudBillingBudgetsV1Budget
      resource to be passed as the request body.
    name: Output only. Resource name of the budget. The resource name implies
      the scope of a budget. Values are of the form
      `billingAccounts/{billingAccountId}/budgets/{budgetId}`.
    updateMask: Optional. Indicates which fields in the provided budget to
      update. Read-only fields (such as `name`) cannot be changed. If this is
      not provided, then only fields with non-default values from the request
      are updated. See https://developers.google.com/protocol-
      buffers/docs/proto3#default for more details about default values.
  """
    googleCloudBillingBudgetsV1Budget = _messages.MessageField('GoogleCloudBillingBudgetsV1Budget', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)