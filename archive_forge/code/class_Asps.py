from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Asps(_messages.Message):
    """A Asps object.

  Fields:
    etag: ETag of the resource.
    items: A list of ASP resources.
    kind: The type of the API resource. This is always
      admin#directory#aspList.
  """
    etag = _messages.StringField(1)
    items = _messages.MessageField('Asp', 2, repeated=True)
    kind = _messages.StringField(3, default=u'admin#directory#aspList')