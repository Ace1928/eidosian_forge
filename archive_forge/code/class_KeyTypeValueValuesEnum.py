from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyTypeValueValuesEnum(_messages.Enum):
    """The type of the key, either stored in `public_key` or referenced in
    `key_id`.

    Values:
      KEY_TYPE_UNSPECIFIED: `KeyType` is not set.
      PGP_ASCII_ARMORED: `PGP ASCII Armored` public key.
      PKIX_PEM: `PKIX PEM` public key.
    """
    KEY_TYPE_UNSPECIFIED = 0
    PGP_ASCII_ARMORED = 1
    PKIX_PEM = 2