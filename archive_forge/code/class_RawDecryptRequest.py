from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RawDecryptRequest(_messages.Message):
    """Request message for KeyManagementService.RawDecrypt.

  Fields:
    additionalAuthenticatedData: Optional. Optional data that must match the
      data originally supplied in
      RawEncryptRequest.additional_authenticated_data.
    additionalAuthenticatedDataCrc32c: Optional. An optional CRC32C checksum
      of the RawDecryptRequest.additional_authenticated_data. If specified,
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
    ciphertext: Required. The encrypted data originally returned in
      RawEncryptResponse.ciphertext.
    ciphertextCrc32c: Optional. An optional CRC32C checksum of the
      RawDecryptRequest.ciphertext. If specified, KeyManagementService will
      verify the integrity of the received ciphertext using this checksum.
      KeyManagementService will report an error if the checksum verification
      fails. If you receive a checksum error, your client should verify that
      CRC32C(ciphertext) is equal to ciphertext_crc32c, and if so, perform a
      limited number of retries. A persistent mismatch may indicate an issue
      in your computation of the CRC32C checksum. Note: This field is defined
      as int64 for reasons of compatibility across different languages.
      However, it is a non-negative integer, which will never exceed 2^32-1,
      and can be safely downconverted to uint32 in languages that support this
      type.
    initializationVector: Required. The initialization vector (IV) used during
      encryption, which must match the data originally provided in
      RawEncryptResponse.initialization_vector.
    initializationVectorCrc32c: Optional. An optional CRC32C checksum of the
      RawDecryptRequest.initialization_vector. If specified,
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
    tagLength: The length of the authentication tag that is appended to the
      end of the ciphertext. If unspecified (0), the default value for the
      key's algorithm will be used (for AES-GCM, the default value is 16).
  """
    additionalAuthenticatedData = _messages.BytesField(1)
    additionalAuthenticatedDataCrc32c = _messages.IntegerField(2)
    ciphertext = _messages.BytesField(3)
    ciphertextCrc32c = _messages.IntegerField(4)
    initializationVector = _messages.BytesField(5)
    initializationVectorCrc32c = _messages.IntegerField(6)
    tagLength = _messages.IntegerField(7, variant=_messages.Variant.INT32)