from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApproveGrantRequest(_messages.Message):
    """Request message for `ApproveGrant` method.

  Fields:
    reason: Optional. The reason for approving this Grant. This is required if
      `require_approver_justification` field of the ManualApprovals workflow
      used in this Grant is true.
  """
    reason = _messages.StringField(1)