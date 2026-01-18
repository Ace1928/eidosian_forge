from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomerEncryptionValue(_messages.Message):
    """Metadata of customer-supplied encryption key, if the object is
    encrypted by such a key.

    Fields:
      encryptionAlgorithm: The encryption algorithm.
      keySha256: SHA256 hash value of the encryption key.
    """
    encryptionAlgorithm = _messages.StringField(1)
    keySha256 = _messages.StringField(2)