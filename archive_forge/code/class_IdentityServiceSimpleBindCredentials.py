from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityServiceSimpleBindCredentials(_messages.Message):
    """The structure holds the LDAP simple binding credential.

  Fields:
    dn: Required. The distinguished name(DN) of the service account
      object/user.
    encryptedPassword: Output only. The encrypted password of the service
      account object/user.
    password: Required. Input only. The password of the service account
      object/user.
  """
    dn = _messages.StringField(1)
    encryptedPassword = _messages.BytesField(2)
    password = _messages.StringField(3)