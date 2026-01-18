from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Asp(_messages.Message):
    """The template that returns individual ASP (Access Code) data.

  Fields:
    codeId: The unique ID of the ASP.
    creationTime: The time when the ASP was created. Expressed in Unix time
      format.
    etag: ETag of the ASP.
    kind: The type of the API resource. This is always admin#directory#asp.
    lastTimeUsed: The time when the ASP was last used. Expressed in Unix time
      format.
    name: The name of the application that the user, represented by their
      userId, entered when the ASP was created.
    userKey: The unique ID of the user who issued the ASP.
  """
    codeId = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    creationTime = _messages.IntegerField(2)
    etag = _messages.StringField(3)
    kind = _messages.StringField(4, default=u'admin#directory#asp')
    lastTimeUsed = _messages.IntegerField(5)
    name = _messages.StringField(6)
    userKey = _messages.StringField(7)