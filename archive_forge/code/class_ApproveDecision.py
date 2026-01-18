from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ApproveDecision(_messages.Message):
    """A decision that has been made to approve access to a resource.

  Fields:
    approveTime: The time at which approval was granted.
    autoApproved: True when the request has been auto-approved.
    expireTime: The time at which the approval expires.
    invalidateTime: If set, denotes the timestamp at which the approval is
      invalidated.
    signatureInfo: The signature for the ApprovalRequest and details on how it
      was signed.
  """
    approveTime = _messages.StringField(1)
    autoApproved = _messages.BooleanField(2)
    expireTime = _messages.StringField(3)
    invalidateTime = _messages.StringField(4)
    signatureInfo = _messages.MessageField('SignatureInfo', 5)