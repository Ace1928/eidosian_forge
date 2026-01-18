from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildApproval(_messages.Message):
    """BuildApproval describes a build's approval configuration, state, and
  result.

  Enums:
    StateValueValuesEnum: Output only. The state of this build's approval.

  Fields:
    config: Output only. Configuration for manual approval of this build.
    result: Output only. Result of manual approval for this Build.
    state: Output only. The state of this build's approval.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of this build's approval.

    Values:
      STATE_UNSPECIFIED: Default enum type. This should not be used.
      PENDING: Build approval is pending.
      APPROVED: Build approval has been approved.
      REJECTED: Build approval has been rejected.
      CANCELLED: Build was cancelled while it was still pending approval.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        APPROVED = 2
        REJECTED = 3
        CANCELLED = 4
    config = _messages.MessageField('ApprovalConfig', 1)
    result = _messages.MessageField('ApprovalResult', 2)
    state = _messages.EnumField('StateValueValuesEnum', 3)