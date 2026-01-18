from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdpCredential(_messages.Message):
    """Credential for verifying signatures produced by the Identity Provider.

  Fields:
    dsaKeyInfo: Output only. Information of a DSA public key.
    name: Output only. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      credential.
    rsaKeyInfo: Output only. Information of a RSA public key.
    updateTime: Output only. Time when the `IdpCredential` was last updated.
  """
    dsaKeyInfo = _messages.MessageField('DsaPublicKeyInfo', 1)
    name = _messages.StringField(2)
    rsaKeyInfo = _messages.MessageField('RsaPublicKeyInfo', 3)
    updateTime = _messages.StringField(4)