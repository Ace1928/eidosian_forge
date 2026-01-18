from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingFoldersLogsDeleteRequest(_messages.Message):
    """A LoggingFoldersLogsDeleteRequest object.

  Fields:
    logName: Required. The resource name of the log to delete:
      projects/[PROJECT_ID]/logs/[LOG_ID]
      organizations/[ORGANIZATION_ID]/logs/[LOG_ID]
      billingAccounts/[BILLING_ACCOUNT_ID]/logs/[LOG_ID]
      folders/[FOLDER_ID]/logs/[LOG_ID][LOG_ID] must be URL-encoded. For
      example, "projects/my-project-id/logs/syslog",
      "organizations/123/logs/cloudaudit.googleapis.com%2Factivity".For more
      information about log names, see LogEntry.
  """
    logName = _messages.StringField(1, required=True)