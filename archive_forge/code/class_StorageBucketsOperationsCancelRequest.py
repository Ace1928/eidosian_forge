from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBucketsOperationsCancelRequest(_messages.Message):
    """A StorageBucketsOperationsCancelRequest object.

  Fields:
    bucket: The parent bucket of the operation resource.
    operationId: The ID of the operation resource.
  """
    bucket = _messages.StringField(1, required=True)
    operationId = _messages.StringField(2, required=True)