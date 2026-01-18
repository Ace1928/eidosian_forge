from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingOrganizationsSinksPatchRequest(_messages.Message):
    """A LoggingOrganizationsSinksPatchRequest object.

  Fields:
    customWriterIdentity: Optional. A service account provided by the caller
      that will be used to write the log entries. The format must be
      serviceAccount:some@email. This field can only be specified if you are
      routing logs to a destination outside this sink's project. If not
      specified, a Logging service account will automatically be generated.
    logSink: A LogSink resource to be passed as the request body.
    sinkName: Required. The full resource name of the sink to update,
      including the parent resource and the sink identifier:
      "projects/[PROJECT_ID]/sinks/[SINK_ID]"
      "organizations/[ORGANIZATION_ID]/sinks/[SINK_ID]"
      "billingAccounts/[BILLING_ACCOUNT_ID]/sinks/[SINK_ID]"
      "folders/[FOLDER_ID]/sinks/[SINK_ID]" For example:"projects/my-
      project/sinks/my-sink"
    uniqueWriterIdentity: Optional. See sinks.create for a description of this
      field. When updating a sink, the effect of this field on the value of
      writer_identity in the updated sink depends on both the old and new
      values of this field: If the old and new values of this field are both
      false or both true, then there is no change to the sink's
      writer_identity. If the old value is false and the new value is true,
      then writer_identity is changed to a service agent
      (https://cloud.google.com/iam/docs/service-account-types#service-agents)
      owned by Cloud Logging. It is an error if the old value is true and the
      new value is set to false or defaulted to false.
    updateMask: Optional. Field mask that specifies the fields in sink that
      need an update. A sink field will be overwritten if, and only if, it is
      in the update mask. name and output only fields cannot be updated.An
      empty updateMask is temporarily treated as using the following mask for
      backwards compatibility purposes:destination,filter,includeChildrenAt
      some point in the future, behavior will be removed and specifying an
      empty updateMask will be an error.For a detailed FieldMask definition,
      see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#google.protobuf.FieldMaskFor
      example: updateMask=filter
  """
    customWriterIdentity = _messages.StringField(1)
    logSink = _messages.MessageField('LogSink', 2)
    sinkName = _messages.StringField(3, required=True)
    uniqueWriterIdentity = _messages.BooleanField(4)
    updateMask = _messages.StringField(5)