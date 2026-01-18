from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1mainPartnerServiceOperationMetadata(_messages.Message):
    """Represents the metadata of the long-running operation.

  Fields:
    apiVersion: Output only. API version used to start the operation.
    createTime: Output only. The time the operation was created.
    endTime: Output only. The time the operation finished running.
    requestedCancellation: Output only. Identifies whether the caller has
      requested cancellation of the operation. Operations that have
      successfully been cancelled have Operation.error value with a
      google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.
    statusMessage: Output only. Human-readable status of the operation, if
      any.
    target: Output only. Server-defined resource path for the target of the
      operation.
    verb: Output only. Name of the verb executed by the operation.
  """
    apiVersion = _messages.StringField(1)
    createTime = _messages.StringField(2)
    endTime = _messages.StringField(3)
    requestedCancellation = _messages.BooleanField(4)
    statusMessage = _messages.StringField(5)
    target = _messages.StringField(6)
    verb = _messages.StringField(7)