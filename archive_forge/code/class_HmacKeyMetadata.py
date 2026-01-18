from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HmacKeyMetadata(_messages.Message):
    """An HMAC Key Metadata resource.

  Fields:
    id: The ID of the HMAC key, including the Project ID and the Access ID.
    selfLink: The link to this resource.
    accessId: The ID of the HMAC Key.
    projectId: Project ID owning the service account to which the key
       authenticates.
    serviceAccountEmail: The email address of the key's associated service
      account.
    state: The state of the key. Can be one of ACTIVE, INACTIVE, or DELETED.
    timeCreated: The creation time of the key in RFC 3339 format.
    updated: The modification time of the key in RFC 3339 format.
    etag: HTTP 1.1 Entity tag for the key.
    kind: The kind of item this is. For HMAC Key metadata, this is always
      storage#hmacKeyMetadata
  """
    id = _messages.StringField(1)
    selfLink = _messages.StringField(2)
    accessId = _messages.StringField(3)
    projectId = _messages.StringField(4)
    serviceAccountEmail = _messages.StringField(5)
    state = _messages.StringField(6)
    timeCreated = _message_types.DateTimeField(7)
    updated = _message_types.DateTimeField(8)
    etag = _messages.StringField(9)
    kind = _messages.StringField(10, default=u'storage#hmacKeyMetadata')