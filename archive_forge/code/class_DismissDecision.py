from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DismissDecision(_messages.Message):
    """A decision that has been made to dismiss an approval request.

  Fields:
    dismissTime: The time at which the approval request was dismissed.
    implicit: This field will be true if the ApprovalRequest was implicitly
      dismissed due to inaction by the access approval approvers (the request
      is not acted on by the approvers before the exiration time).
  """
    dismissTime = _messages.StringField(1)
    implicit = _messages.BooleanField(2)