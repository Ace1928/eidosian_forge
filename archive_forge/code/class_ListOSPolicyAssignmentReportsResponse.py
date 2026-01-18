from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListOSPolicyAssignmentReportsResponse(_messages.Message):
    """A response message for listing OS Policy assignment reports including
  the page of results and page token.

  Fields:
    nextPageToken: The pagination token to retrieve the next page of OS policy
      assignment report objects.
    osPolicyAssignmentReports: List of OS policy assignment reports.
  """
    nextPageToken = _messages.StringField(1)
    osPolicyAssignmentReports = _messages.MessageField('OSPolicyAssignmentReport', 2, repeated=True)