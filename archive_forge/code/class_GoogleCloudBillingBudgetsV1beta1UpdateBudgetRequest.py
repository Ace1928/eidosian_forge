from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBillingBudgetsV1beta1UpdateBudgetRequest(_messages.Message):
    """Request for UpdateBudget

  Fields:
    budget: Required. The updated budget object. The budget to update is
      specified by the budget name in the budget.
    updateMask: Optional. Indicates which fields in the provided budget to
      update. Read-only fields (such as `name`) cannot be changed. If this is
      not provided, then only fields with non-default values from the request
      are updated. See https://developers.google.com/protocol-
      buffers/docs/proto3#default for more details about default values.
  """
    budget = _messages.MessageField('GoogleCloudBillingBudgetsV1beta1Budget', 1)
    updateMask = _messages.StringField(2)