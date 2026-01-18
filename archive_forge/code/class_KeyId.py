from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyId(_messages.Message):
    """A KeyId identifies a specific public key, usually by hashing the public
  key.

  Fields:
    keyId: Optional. The value of this KeyId encoded in lowercase hexadecimal.
      This is most likely the 160 bit SHA-1 hash of the public key.
  """
    keyId = _messages.StringField(1)