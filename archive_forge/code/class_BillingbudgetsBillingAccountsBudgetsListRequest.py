from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BillingbudgetsBillingAccountsBudgetsListRequest(_messages.Message):
    """A BillingbudgetsBillingAccountsBudgetsListRequest object.

  Fields:
    pageSize: Optional. The maximum number of budgets to return per page. The
      default and maximum value are 100.
    pageToken: Optional. The value returned by the last `ListBudgetsResponse`
      which indicates that this is a continuation of a prior `ListBudgets`
      call, and that the system should return the next page of data.
    parent: Required. Name of billing account to list budgets under. Values
      are of the form `billingAccounts/{billingAccountId}`.
    scope: Optional. Set the scope of the budgets to be returned, in the
      format of the resource name. The scope of a budget is the cost that it
      tracks, such as costs for a single project, or the costs for all
      projects in a folder. Only project scope (in the format of
      "projects/project-id" or "projects/123") is supported in this field.
      When this field is set to a project's resource name, the budgets
      returned are tracking the costs for that project.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    scope = _messages.StringField(4)