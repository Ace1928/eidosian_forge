from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListAssignmentsResponse(_messages.Message):
    """A response message for listing Assignments.

  Fields:
    assignments: The list of Assignments.
    nextPageToken: A pagination token that can be used to get the next page of
      Assignments.
  """
    assignments = _messages.MessageField('Assignment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)