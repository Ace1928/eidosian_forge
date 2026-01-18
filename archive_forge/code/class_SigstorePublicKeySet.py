from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SigstorePublicKeySet(_messages.Message):
    """A bundle of Sigstore public keys, used to verify Sigstore signatures. A
  signature is authenticated by a `SigstorePublicKeySet` if any of the keys
  verify it.

  Fields:
    publicKeys: Required. `public_keys` must have at least one entry.
  """
    publicKeys = _messages.MessageField('SigstorePublicKey', 1, repeated=True)