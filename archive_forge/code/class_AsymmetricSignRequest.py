from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AsymmetricSignRequest(_messages.Message):
    """Request message for KeyManagementService.AsymmetricSign.

  Fields:
    data: Optional. The data to sign. It can't be supplied if
      AsymmetricSignRequest.digest is supplied.
    dataCrc32c: Optional. An optional CRC32C checksum of the
      AsymmetricSignRequest.data. If specified, KeyManagementService will
      verify the integrity of the received AsymmetricSignRequest.data using
      this checksum. KeyManagementService will report an error if the checksum
      verification fails. If you receive a checksum error, your client should
      verify that CRC32C(AsymmetricSignRequest.data) is equal to
      AsymmetricSignRequest.data_crc32c, and if so, perform a limited number
      of retries. A persistent mismatch may indicate an issue in your
      computation of the CRC32C checksum. Note: This field is defined as int64
      for reasons of compatibility across different languages. However, it is
      a non-negative integer, which will never exceed 2^32-1, and can be
      safely downconverted to uint32 in languages that support this type.
    digest: Optional. The digest of the data to sign. The digest must be
      produced with the same digest algorithm as specified by the key
      version's algorithm. This field may not be supplied if
      AsymmetricSignRequest.data is supplied.
    digestCrc32c: Optional. An optional CRC32C checksum of the
      AsymmetricSignRequest.digest. If specified, KeyManagementService will
      verify the integrity of the received AsymmetricSignRequest.digest using
      this checksum. KeyManagementService will report an error if the checksum
      verification fails. If you receive a checksum error, your client should
      verify that CRC32C(AsymmetricSignRequest.digest) is equal to
      AsymmetricSignRequest.digest_crc32c, and if so, perform a limited number
      of retries. A persistent mismatch may indicate an issue in your
      computation of the CRC32C checksum. Note: This field is defined as int64
      for reasons of compatibility across different languages. However, it is
      a non-negative integer, which will never exceed 2^32-1, and can be
      safely downconverted to uint32 in languages that support this type.
  """
    data = _messages.BytesField(1)
    dataCrc32c = _messages.IntegerField(2)
    digest = _messages.MessageField('Digest', 3)
    digestCrc32c = _messages.IntegerField(4)