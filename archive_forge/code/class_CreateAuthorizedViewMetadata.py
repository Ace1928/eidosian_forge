from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateAuthorizedViewMetadata(_messages.Message):
    """The metadata for the Operation returned by CreateAuthorizedView.

  Fields:
    finishTime: The time at which the operation failed or was completed
      successfully.
    originalRequest: The request that prompted the initiation of this
      CreateInstance operation.
    requestTime: The time at which the original request was received.
  """
    finishTime = _messages.StringField(1)
    originalRequest = _messages.MessageField('CreateAuthorizedViewRequest', 2)
    requestTime = _messages.StringField(3)