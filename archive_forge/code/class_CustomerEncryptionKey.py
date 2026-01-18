from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomerEncryptionKey(_messages.Message):
    """A customer-supplied encryption key.

  Fields:
    kmsKeyName: Optional. The name of the encryption key that is stored in
      Google Cloud KMS.
    kmsKeyServiceAccount: Optional. The service account being used for the
      encryption request for the given KMS key. If absent, the Compute Engine
      default service account is used.
    rawKey: Optional. Specifies a 256-bit customer-supplied encryption key.
    rsaEncryptedKey: Optional. RSA-wrapped 2048-bit customer-supplied
      encryption key to either encrypt or decrypt this resource.
  """
    kmsKeyName = _messages.StringField(1)
    kmsKeyServiceAccount = _messages.StringField(2)
    rawKey = _messages.StringField(3)
    rsaEncryptedKey = _messages.StringField(4)