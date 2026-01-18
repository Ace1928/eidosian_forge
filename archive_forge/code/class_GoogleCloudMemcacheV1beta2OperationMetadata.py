from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMemcacheV1beta2OperationMetadata(_messages.Message):
    """Represents the metadata of a long-running operation.

  Fields:
    apiVersion: Output only. API version used to start the operation.
    cancelRequested: Output only. Identifies whether the user has requested
      cancellation of the operation. Operations that have successfully been
      cancelled have Operation.error value with a google.rpc.Status.code of 1,
      corresponding to `Code.CANCELLED`.
    createTime: Output only. Time when the operation was created.
    endTime: Output only. Time when the operation finished running.
    statusDetail: Output only. Human-readable status of the operation, if any.
    target: Output only. Server-defined resource path for the target of the
      operation.
    verb: Output only. Name of the verb executed by the operation.
  """
    apiVersion = _messages.StringField(1)
    cancelRequested = _messages.BooleanField(2)
    createTime = _messages.StringField(3)
    endTime = _messages.StringField(4)
    statusDetail = _messages.StringField(5)
    target = _messages.StringField(6)
    verb = _messages.StringField(7)