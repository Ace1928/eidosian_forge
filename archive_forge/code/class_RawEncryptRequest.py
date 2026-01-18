from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RawEncryptRequest(_messages.Message):
    """Request message for KeyManagementService.RawEncrypt.

  Fields:
    additionalAuthenticatedData: Optional. Optional data that, if specified,
      must also be provided during decryption through
      RawDecryptRequest.additional_authenticated_data. This field may only be
      used in conjunction with an algorithm that accepts additional
      authenticated data (for example, AES-GCM). The maximum size depends on
      the key version's protection_level. For SOFTWARE keys, the plaintext
      must be no larger than 64KiB. For HSM keys, the combined length of the
      plaintext and additional_authenticated_data fields must be no larger
      than 8KiB.
    additionalAuthenticatedDataCrc32c: Optional. An optional CRC32C checksum
      of the RawEncryptRequest.additional_authenticated_data. If specified,
      KeyManagementService will verify the integrity of the received
      additional_authenticated_data using this checksum. KeyManagementService
      will report an error if the checksum verification fails. If you receive
      a checksum error, your client should verify that
      CRC32C(additional_authenticated_data) is equal to
      additional_authenticated_data_crc32c, and if so, perform a limited
      number of retries. A persistent mismatch may indicate an issue in your
      computation of the CRC32C checksum. Note: This field is defined as int64
      for reasons of compatibility across different languages. However, it is
      a non-negative integer, which will never exceed 2^32-1, and can be
      safely downconverted to uint32 in languages that support this type.
    initializationVector: Optional. A customer-supplied initialization vector
      that will be used for encryption. If it is not provided for AES-CBC and
      AES-CTR, one will be generated. It will be returned in
      RawEncryptResponse.initialization_vector.
    initializationVectorCrc32c: Optional. An optional CRC32C checksum of the
      RawEncryptRequest.initialization_vector. If specified,
      KeyManagementService will verify the integrity of the received
      initialization_vector using this checksum. KeyManagementService will
      report an error if the checksum verification fails. If you receive a
      checksum error, your client should verify that
      CRC32C(initialization_vector) is equal to initialization_vector_crc32c,
      and if so, perform a limited number of retries. A persistent mismatch
      may indicate an issue in your computation of the CRC32C checksum. Note:
      This field is defined as int64 for reasons of compatibility across
      different languages. However, it is a non-negative integer, which will
      never exceed 2^32-1, and can be safely downconverted to uint32 in
      languages that support this type.
    plaintext: Required. The data to encrypt. Must be no larger than 64KiB.
      The maximum size depends on the key version's protection_level. For
      SOFTWARE keys, the plaintext must be no larger than 64KiB. For HSM keys,
      the combined length of the plaintext and additional_authenticated_data
      fields must be no larger than 8KiB.
    plaintextCrc32c: Optional. An optional CRC32C checksum of the
      RawEncryptRequest.plaintext. If specified, KeyManagementService will
      verify the integrity of the received plaintext using this checksum.
      KeyManagementService will report an error if the checksum verification
      fails. If you receive a checksum error, your client should verify that
      CRC32C(plaintext) is equal to plaintext_crc32c, and if so, perform a
      limited number of retries. A persistent mismatch may indicate an issue
      in your computation of the CRC32C checksum. Note: This field is defined
      as int64 for reasons of compatibility across different languages.
      However, it is a non-negative integer, which will never exceed 2^32-1,
      and can be safely downconverted to uint32 in languages that support this
      type.
  """
    additionalAuthenticatedData = _messages.BytesField(1)
    additionalAuthenticatedDataCrc32c = _messages.IntegerField(2)
    initializationVector = _messages.BytesField(3)
    initializationVectorCrc32c = _messages.IntegerField(4)
    plaintext = _messages.BytesField(5)
    plaintextCrc32c = _messages.IntegerField(6)