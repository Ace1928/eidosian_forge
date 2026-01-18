from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildSignature(_messages.Message):
    """Message encapsulating the signature of the verified build.

  Enums:
    KeyTypeValueValuesEnum: The type of the key, either stored in `public_key`
      or referenced in `key_id`.

  Fields:
    keyId: An ID for the key used to sign. This could be either an ID for the
      key stored in `public_key` (such as the ID or fingerprint for a PGP key,
      or the CN for a cert), or a reference to an external key (such as a
      reference to a key in Cloud Key Management Service).
    keyType: The type of the key, either stored in `public_key` or referenced
      in `key_id`.
    publicKey: Public key of the builder which can be used to verify that the
      related findings are valid and unchanged. If `key_type` is empty, this
      defaults to PEM encoded public keys. This field may be empty if `key_id`
      references an external key. For Cloud Build based signatures, this is a
      PEM encoded public key. To verify the Cloud Build signature, place the
      contents of this field into a file (public.pem). The signature field is
      base64-decoded into its binary representation in signature.bin, and the
      provenance bytes from `BuildDetails` are base64-decoded into a binary
      representation in signed.bin. OpenSSL can then verify the signature:
      `openssl sha256 -verify public.pem -signature signature.bin signed.bin`
    signature: Required. Signature of the related `BuildProvenance`. In JSON,
      this is base-64 encoded.
  """

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
    keyId = _messages.StringField(1)
    keyType = _messages.EnumField('KeyTypeValueValuesEnum', 2)
    publicKey = _messages.StringField(3)
    signature = _messages.BytesField(4)