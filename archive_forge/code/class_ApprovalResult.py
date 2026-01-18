from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApprovalResult(_messages.Message):
    """ApprovalResult describes the decision and associated metadata of a
  manual approval of a build.

  Enums:
    DecisionValueValuesEnum: Required. The decision of this manual approval.

  Fields:
    approvalTime: Output only. The time when the approval decision was made.
    approverAccount: Output only. Email of the user that called the
      ApproveBuild API to approve or reject a build at the time that the API
      was called.
    comment: Optional. An optional comment for this manual approval result.
    decision: Required. The decision of this manual approval.
    url: Optional. An optional URL tied to this manual approval result. This
      field is essentially the same as comment, except that it will be
      rendered by the UI differently. An example use case is a link to an
      external job that approved this Build.
  """

    class DecisionValueValuesEnum(_messages.Enum):
        """Required. The decision of this manual approval.

    Values:
      DECISION_UNSPECIFIED: Default enum type. This should not be used.
      APPROVED: Build is approved.
      REJECTED: Build is rejected.
    """
        DECISION_UNSPECIFIED = 0
        APPROVED = 1
        REJECTED = 2
    approvalTime = _messages.StringField(1)
    approverAccount = _messages.StringField(2)
    comment = _messages.StringField(3)
    decision = _messages.EnumField('DecisionValueValuesEnum', 4)
    url = _messages.StringField(5)