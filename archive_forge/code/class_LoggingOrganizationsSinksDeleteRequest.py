from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingOrganizationsSinksDeleteRequest(_messages.Message):
    """A LoggingOrganizationsSinksDeleteRequest object.

  Fields:
    sinkName: Required. The full resource name of the sink to delete,
      including the parent resource and the sink identifier:
      "projects/[PROJECT_ID]/sinks/[SINK_ID]"
      "organizations/[ORGANIZATION_ID]/sinks/[SINK_ID]"
      "billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_ID]"
      "folders/[FOLDER_ID]/sinks/[SINK_ID]" For example:"projects/my-
      project/sinks/my-sink"
  """
    sinkName = _messages.StringField(1, required=True)