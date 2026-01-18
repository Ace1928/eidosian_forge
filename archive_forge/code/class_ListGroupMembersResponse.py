from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListGroupMembersResponse(_messages.Message):
    """The response message of the `ListGroupMembers` method.

  Fields:
    memberStates: The member states exposed by the parent group.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    memberStates = _messages.MessageField('MemberState', 1, repeated=True)
    nextPageToken = _messages.StringField(2)