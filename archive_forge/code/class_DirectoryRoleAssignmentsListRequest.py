from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryRoleAssignmentsListRequest(_messages.Message):
    """A DirectoryRoleAssignmentsListRequest object.

  Fields:
    customer: Immutable ID of the G Suite account.
    maxResults: Maximum number of results to return.
    pageToken: Token to specify the next page in the list.
    roleId: Immutable ID of a role. If included in the request, returns only
      role assignments containing this role ID.
    userKey: The user's primary email address, alias email address, or unique
      user ID. If included in the request, returns role assignments only for
      this user.
  """
    customer = _messages.StringField(1, required=True)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    roleId = _messages.StringField(4)
    userKey = _messages.StringField(5)