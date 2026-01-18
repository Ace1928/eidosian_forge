from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CustomAccount(_messages.Message):
    """Describes authentication configuration that uses a custom account.

  Fields:
    loginUrl: Required. The login form URL of the website.
    password: Required. Input only. The password of the custom account. The
      credential is stored encrypted and not returned in any response nor
      included in audit logs.
    username: Required. The user name of the custom account.
  """
    loginUrl = _messages.StringField(1)
    password = _messages.StringField(2)
    username = _messages.StringField(3)