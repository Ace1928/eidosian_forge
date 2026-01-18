from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MacSignResponse(_messages.Message):
    """Response message for KeyManagementService.MacSign.

  Enums:
    ProtectionLevelValueValuesEnum: The ProtectionLevel of the
      CryptoKeyVersion used for signing.

  Fields:
    mac: The created signature.
    macCrc32c: Integrity verification field. A CRC32C checksum of the returned
      MacSignResponse.mac. An integrity check of MacSignResponse.mac can be
      performed by computing the CRC32C checksum of MacSignResponse.mac and
      comparing your results to this field. Discard the response in case of
      non-matching checksum values, and perform a limited number of retries. A
      persistent mismatch may indicate an issue in your computation of the
      CRC32C checksum. Note: This field is defined as int64 for reasons of
      compatibility across different languages. However, it is a non-negative
      integer, which will never exceed 2^32-1, and can be safely downconverted
      to uint32 in languages that support this type.
    name: The resource name of the CryptoKeyVersion used for signing. Check
      this field to verify that the intended resource was used for signing.
    protectionLevel: The ProtectionLevel of the CryptoKeyVersion used for
      signing.
    verifiedDataCrc32c: Integrity verification field. A flag indicating
      whether MacSignRequest.data_crc32c was received by KeyManagementService
      and used for the integrity verification of the data. A false value of
      this field indicates either that MacSignRequest.data_crc32c was left
      unset or that it was not delivered to KeyManagementService. If you've
      set MacSignRequest.data_crc32c but this field is still false, discard
      the response and perform a limited number of retries.
  """

    class ProtectionLevelValueValuesEnum(_messages.Enum):
        """The ProtectionLevel of the CryptoKeyVersion used for signing.

    Values:
      PROTECTION_LEVEL_UNSPECIFIED: Not specified.
      SOFTWARE: Crypto operations are performed in software.
      HSM: Crypto operations are performed in a Hardware Security Module.
      EXTERNAL: Crypto operations are performed by an external key manager.
      EXTERNAL_VPC: Crypto operations are performed in an EKM-over-VPC
        backend.
    """
        PROTECTION_LEVEL_UNSPECIFIED = 0
        SOFTWARE = 1
        HSM = 2
        EXTERNAL = 3
        EXTERNAL_VPC = 4
    mac = _messages.BytesField(1)
    macCrc32c = _messages.IntegerField(2)
    name = _messages.StringField(3)
    protectionLevel = _messages.EnumField('ProtectionLevelValueValuesEnum', 4)
    verifiedDataCrc32c = _messages.BooleanField(5)