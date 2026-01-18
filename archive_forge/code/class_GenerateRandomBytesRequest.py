from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateRandomBytesRequest(_messages.Message):
    """Request message for KeyManagementService.GenerateRandomBytes.

  Enums:
    ProtectionLevelValueValuesEnum: The ProtectionLevel to use when generating
      the random data. Currently, only HSM protection level is supported.

  Fields:
    lengthBytes: The length in bytes of the amount of randomness to retrieve.
      Minimum 8 bytes, maximum 1024 bytes.
    protectionLevel: The ProtectionLevel to use when generating the random
      data. Currently, only HSM protection level is supported.
  """

    class ProtectionLevelValueValuesEnum(_messages.Enum):
        """The ProtectionLevel to use when generating the random data. Currently,
    only HSM protection level is supported.

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
    lengthBytes = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    protectionLevel = _messages.EnumField('ProtectionLevelValueValuesEnum', 2)