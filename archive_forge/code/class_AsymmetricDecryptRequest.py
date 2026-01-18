from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AsymmetricDecryptRequest(_messages.Message):
    """Request message for KeyManagementService.AsymmetricDecrypt.

  Fields:
    ciphertext: Required. The data encrypted with the named CryptoKeyVersion's
      public key using OAEP.
    ciphertextCrc32c: Optional. An optional CRC32C checksum of the
      AsymmetricDecryptRequest.ciphertext. If specified, KeyManagementService
      will verify the integrity of the received
      AsymmetricDecryptRequest.ciphertext using this checksum.
      KeyManagementService will report an error if the checksum verification
      fails. If you receive a checksum error, your client should verify that
      CRC32C(AsymmetricDecryptRequest.ciphertext) is equal to
      AsymmetricDecryptRequest.ciphertext_crc32c, and if so, perform a limited
      number of retries. A persistent mismatch may indicate an issue in your
      computation of the CRC32C checksum. Note: This field is defined as int64
      for reasons of compatibility across different languages. However, it is
      a non-negative integer, which will never exceed 2^32-1, and can be
      safely downconverted to uint32 in languages that support this type.
  """
    ciphertext = _messages.BytesField(1)
    ciphertextCrc32c = _messages.IntegerField(2)