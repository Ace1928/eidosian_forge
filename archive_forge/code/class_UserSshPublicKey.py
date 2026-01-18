from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserSshPublicKey(_messages.Message):
    """JSON template for a POSIX account entry.

  Fields:
    expirationTimeUsec: An expiration time in microseconds since epoch.
    fingerprint: A SHA-256 fingerprint of the SSH public key. (Read-only)
    key: An SSH public key.
  """
    expirationTimeUsec = _messages.IntegerField(1)
    fingerprint = _messages.StringField(2)
    key = _messages.StringField(3)