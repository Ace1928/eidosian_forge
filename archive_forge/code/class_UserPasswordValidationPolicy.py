from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class UserPasswordValidationPolicy(_messages.Message):
    """User level password validation policy.

  Fields:
    allowedFailedAttempts: Number of failed login attempts allowed before user
      get locked.
    enableFailedAttemptsCheck: If true, failed login attempts check will be
      enabled.
    enablePasswordVerification: If true, the user must specify the current
      password before changing the password. This flag is supported only for
      MySQL.
    passwordExpirationDuration: Expiration duration after password is updated.
    status: Output only. Read-only password status.
  """
    allowedFailedAttempts = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    enableFailedAttemptsCheck = _messages.BooleanField(2)
    enablePasswordVerification = _messages.BooleanField(3)
    passwordExpirationDuration = _messages.StringField(4)
    status = _messages.MessageField('PasswordStatus', 5)