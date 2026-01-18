from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PurposeValueValuesEnum(_messages.Enum):
    """The immutable purpose of this CryptoKey. Currently, the only acceptable
    purpose is ENCRYPT_DECRYPT.

    Values:
      CRYPTO_KEY_PURPOSE_UNSPECIFIED: Not specified.
      ENCRYPT_DECRYPT: CryptoKeys with this purpose may be used with Encrypt
        and Decrypt.
    """
    CRYPTO_KEY_PURPOSE_UNSPECIFIED = 0
    ENCRYPT_DECRYPT = 1