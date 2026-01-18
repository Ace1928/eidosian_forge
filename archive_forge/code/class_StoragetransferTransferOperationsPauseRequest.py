from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragetransferTransferOperationsPauseRequest(_messages.Message):
    """A StoragetransferTransferOperationsPauseRequest object.

  Fields:
    name: Required. The name of the transfer operation.
    pauseTransferOperationRequest: A PauseTransferOperationRequest resource to
      be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    pauseTransferOperationRequest = _messages.MessageField('PauseTransferOperationRequest', 2)