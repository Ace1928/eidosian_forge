from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Tokens(_messages.Message):
    """JSON response template for List tokens operation in Directory API.

  Fields:
    etag: ETag of the resource.
    items: A list of Token resources.
    kind: The type of the API resource. This is always
      admin#directory#tokenList.
  """
    etag = _messages.StringField(1)
    items = _messages.MessageField('Token', 2, repeated=True)
    kind = _messages.StringField(3, default=u'admin#directory#tokenList')