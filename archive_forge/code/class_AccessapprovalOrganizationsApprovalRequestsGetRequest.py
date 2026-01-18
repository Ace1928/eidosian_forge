from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessapprovalOrganizationsApprovalRequestsGetRequest(_messages.Message):
    """A AccessapprovalOrganizationsApprovalRequestsGetRequest object.

  Fields:
    name: The name of the approval request to retrieve. Format: "{projects|fol
      ders|organizations}/{id}/approvalRequests/{approval_request}"
  """
    name = _messages.StringField(1, required=True)