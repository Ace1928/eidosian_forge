from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SignSshPublicKeyResponse(_messages.Message):
    """A SignSshPublicKeyResponse object.

  Fields:
    signedSshPublicKey: The signed SSH public key to use in the SSH handshake.
  """
    signedSshPublicKey = _messages.StringField(1)