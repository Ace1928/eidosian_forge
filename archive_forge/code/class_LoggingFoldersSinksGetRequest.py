from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingFoldersSinksGetRequest(_messages.Message):
    """A LoggingFoldersSinksGetRequest object.

  Fields:
    sinkName: Required. The resource name of the sink:
      "projects/[PROJECT_ID]/sinks/[SINK_ID]"
      "organizations/[ORGANIZATION_ID]/sinks/[SINK_ID]"
      "billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_ID]"
      "folders/[FOLDER_ID]/sinks/[SINK_ID]" For example:"projects/my-
      project/sinks/my-sink"
  """
    sinkName = _messages.StringField(1, required=True)