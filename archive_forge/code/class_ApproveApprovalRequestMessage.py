from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ApproveApprovalRequestMessage(_messages.Message):
    """Request to approve an ApprovalRequest.

  Fields:
    expireTime: The expiration time of this approval.
  """
    expireTime = _messages.StringField(1)