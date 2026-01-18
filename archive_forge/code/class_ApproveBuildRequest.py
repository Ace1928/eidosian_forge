from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApproveBuildRequest(_messages.Message):
    """Request to approve or reject a pending build.

  Fields:
    approvalResult: Approval decision and metadata.
  """
    approvalResult = _messages.MessageField('ApprovalResult', 1)