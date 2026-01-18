from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PasswordStatus(_messages.Message):
    """Read-only password status.

  Fields:
    locked: If true, user does not have login privileges.
    passwordExpirationTime: The expiration time of the current password.
  """
    locked = _messages.BooleanField(1)
    passwordExpirationTime = _messages.StringField(2)