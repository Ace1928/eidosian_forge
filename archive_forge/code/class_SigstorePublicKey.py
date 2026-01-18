from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SigstorePublicKey(_messages.Message):
    """A Sigstore public key. `SigstorePublicKey` is the public key material
  used to authenticate Sigstore signatures.

  Fields:
    publicKeyPem: The public key material in PEM format.
  """
    publicKeyPem = _messages.StringField(1)