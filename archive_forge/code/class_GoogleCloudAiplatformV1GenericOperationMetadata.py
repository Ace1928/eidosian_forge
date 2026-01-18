from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1GenericOperationMetadata(_messages.Message):
    """Generic Metadata shared by all operations.

  Fields:
    createTime: Output only. Time when the operation was created.
    partialFailures: Output only. Partial failures encountered. E.g. single
      files that couldn't be read. This field should never exceed 20 entries.
      Status details field will contain standard Google Cloud error details.
    updateTime: Output only. Time when the operation was updated for the last
      time. If the operation has finished (successfully or not), this is the
      finish time.
  """
    createTime = _messages.StringField(1)
    partialFailures = _messages.MessageField('GoogleRpcStatus', 2, repeated=True)
    updateTime = _messages.StringField(3)