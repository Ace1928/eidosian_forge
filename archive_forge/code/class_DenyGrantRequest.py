from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DenyGrantRequest(_messages.Message):
    """Request message for `DenyGrant` method.

  Fields:
    reason: Optional. The reason for denying this Grant. This is required if
      `require_approver_justification` field of the ManualApprovals workflow
      used in this Grant is true.
  """
    reason = _messages.StringField(1)