from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListOSPolicyAssignmentsResponse(_messages.Message):
    """A response message for listing all assignments under given parent.

  Fields:
    nextPageToken: The pagination token to retrieve the next page of OS policy
      assignments.
    osPolicyAssignments: The list of assignments
  """
    nextPageToken = _messages.StringField(1)
    osPolicyAssignments = _messages.MessageField('OSPolicyAssignment', 2, repeated=True)