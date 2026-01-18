from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApprovalWorkflow(_messages.Message):
    """Different types of approval workflows that can be used to gate
  privileged access granting.

  Fields:
    manualApprovals: An approval workflow where users designated as approvers
      review and act on the Grants.
  """
    manualApprovals = _messages.MessageField('ManualApprovals', 1)