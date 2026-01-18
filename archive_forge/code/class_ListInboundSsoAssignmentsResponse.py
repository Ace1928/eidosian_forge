from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListInboundSsoAssignmentsResponse(_messages.Message):
    """Response of the InboundSsoAssignmentsService.ListInboundSsoAssignments
  method.

  Fields:
    inboundSsoAssignments: The assignments.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    inboundSsoAssignments = _messages.MessageField('InboundSsoAssignment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)