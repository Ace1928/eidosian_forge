from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingSinksCreateRequest(_messages.Message):
    """A LoggingSinksCreateRequest object.

  Fields:
    customWriterIdentity: Optional. A service account provided by the caller
      that will be used to write the log entries. The format must be
      serviceAccount:some@email. This field can only be specified if you are
      routing logs to a destination outside this sink's project. If not
      specified, a Logging service account will automatically be generated.
    logSink: A LogSink resource to be passed as the request body.
    parent: Required. The resource in which to create the sink:
      "projects/[PROJECT_ID]" "organizations/[ORGANIZATION_ID]"
      "billingAccounts/[BILLING_ACCOUNT_ID]" "folders/[FOLDER_ID]" For
      examples:"projects/my-project" "organizations/123456789"
    uniqueWriterIdentity: Optional. Determines the kind of IAM identity
      returned as writer_identity in the new sink. If this value is omitted or
      set to false, and if the sink's parent is a project, then the value
      returned as writer_identity is the same group or service account used by
      Cloud Logging before the addition of writer identities to this API. The
      sink's destination must be in the same project as the sink itself.If
      this field is set to true, or if the sink is owned by a non-project
      resource such as an organization, then the value of writer_identity will
      be a service agent (https://cloud.google.com/iam/docs/service-account-
      types#service-agents) used by the sinks with the same parent. For more
      information, see writer_identity in LogSink.
  """
    customWriterIdentity = _messages.StringField(1)
    logSink = _messages.MessageField('LogSink', 2)
    parent = _messages.StringField(3, required=True)
    uniqueWriterIdentity = _messages.BooleanField(4)