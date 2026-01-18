from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateAuthorizedViewMetadata(_messages.Message):
    """Metadata for the google.longrunning.Operation returned by
  UpdateAuthorizedView.

  Fields:
    finishTime: The time at which the operation failed or was completed
      successfully.
    originalRequest: The request that prompted the initiation of this
      UpdateAuthorizedView operation.
    requestTime: The time at which the original request was received.
  """
    finishTime = _messages.StringField(1)
    originalRequest = _messages.MessageField('UpdateAuthorizedViewRequest', 2)
    requestTime = _messages.StringField(3)