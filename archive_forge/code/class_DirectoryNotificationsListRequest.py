from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryNotificationsListRequest(_messages.Message):
    """A DirectoryNotificationsListRequest object.

  Fields:
    customer: The unique ID for the customer's G Suite account.
    language: The ISO 639-1 code of the language notifications are returned
      in. The default is English (en).
    maxResults: Maximum number of notifications to return per page. The
      default is 100.
    pageToken: The token to specify the page of results to retrieve.
  """
    customer = _messages.StringField(1, required=True)
    language = _messages.StringField(2)
    maxResults = _messages.IntegerField(3, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(4)