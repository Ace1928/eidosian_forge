from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingBillingAccountsSinksListRequest(_messages.Message):
    """A LoggingBillingAccountsSinksListRequest object.

  Fields:
    filter: Optional. A filter expression to constrain the sinks returned.
      Today, this only supports the following strings: '' 'in_scope("ALL")',
      'in_scope("ANCESTOR")', 'in_scope("DEFAULT")'.Description of scopes
      below. ALL: Includes all of the sinks which can be returned in any other
      scope. ANCESTOR: Includes intercepting sinks owned by ancestor
      resources. DEFAULT: Includes sinks owned by parent.When the empty string
      is provided, then the filter 'in_scope("DEFAULT")' is applied.
    pageSize: Optional. The maximum number of results to return from this
      request. Non-positive values are ignored. The presence of nextPageToken
      in the response indicates that more results might be available.
    pageToken: Optional. If present, then retrieve the next batch of results
      from the preceding call to this method. pageToken must be the value of
      nextPageToken from the previous response. The values of other method
      parameters should be identical to those in the previous call.
    parent: Required. The parent resource whose sinks are to be listed:
      "projects/[PROJECT_ID]" "organizations/[ORGANIZATION_ID]"
      "billingAccounts/[BILLING_ACCOUNT_ID]" "folders/[FOLDER_ID]"
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)