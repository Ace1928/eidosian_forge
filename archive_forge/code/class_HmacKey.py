from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HmacKey(_messages.Message):
    """An HMAC Key resource.

  Fields:
    metadata: Key metadata.
    secret: HMAC secret key material.
    kind: The kind of item this is. For HMAC keys, this is always
      storage#hmacKey.
  """
    metadata = _messages.MessageField('HmacKeyMetadata', 1)
    secret = _messages.StringField(2)
    kind = _messages.StringField(3, default=u'storage#hmacKey')