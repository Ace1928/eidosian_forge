from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Members(_messages.Message):
    """JSON response template for List Members operation in Directory API.

  Fields:
    etag: ETag of the resource.
    kind: Kind of resource this is.
    members: List of member objects.
    nextPageToken: Token used to access next page of this result.
  """
    etag = _messages.StringField(1)
    kind = _messages.StringField(2, default=u'admin#directory#members')
    members = _messages.MessageField('Member', 3, repeated=True)
    nextPageToken = _messages.StringField(4)