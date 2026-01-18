from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryMembersListRequest(_messages.Message):
    """A DirectoryMembersListRequest object.

  Fields:
    groupKey: Email or immutable ID of the group
    includeDerivedMembership: Whether to list indirect memberships. Default:
      false.
    maxResults: Maximum number of results to return. Max allowed value is 200.
    pageToken: Token to specify next page in the list
    roles: Comma separated role values to filter list results on.
  """
    groupKey = _messages.StringField(1, required=True)
    includeDerivedMembership = _messages.BooleanField(2)
    maxResults = _messages.IntegerField(3, variant=_messages.Variant.INT32, default=200)
    pageToken = _messages.StringField(4)
    roles = _messages.StringField(5)