from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProtectionLevelValueValuesEnum(_messages.Enum):
    """The ProtectionLevel of the CryptoKeyVersion used in encryption.

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