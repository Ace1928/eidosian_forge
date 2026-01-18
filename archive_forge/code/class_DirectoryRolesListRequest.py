from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryRolesListRequest(_messages.Message):
    """A DirectoryRolesListRequest object.

  Fields:
    customer: Immutable ID of the G Suite account.
    maxResults: Maximum number of results to return.
    pageToken: Token to specify the next page in the list.
  """
    customer = _messages.StringField(1, required=True)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)