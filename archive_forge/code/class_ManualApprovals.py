from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManualApprovals(_messages.Message):
    """A manual approval workflow where users who are designated as approvers
  need to call the ApproveGrant/DenyGrant APIs for an Grant. The workflow can
  consist of multiple serial steps where each step defines who can act as
  Approver in that step and how many of those users should approve before the
  workflow moves to the next step. This can be used to create approval
  workflows such as * Require an approval from any user in a group G. *
  Require an approval from any k number of users from a Group G. * Require an
  approval from any user in a group G and then from a user U. A single user
  might be part of `approvers` ACL for multiple steps in this workflow but
  they can only approve once and that approval will only be considered to
  satisfy the approval step at which it was granted.

  Fields:
    requireApproverJustification: Optional. Do the approvers need to provide a
      justification for their actions?
    steps: Optional. List of approval steps in this workflow. These steps
      would be followed in the specified order sequentially. Only 1 step is
      supported for now.
  """
    requireApproverJustification = _messages.BooleanField(1)
    steps = _messages.MessageField('Step', 2, repeated=True)