from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyManagementModeValueValuesEnum(_messages.Enum):
    """Optional. Describes who can perform control plane operations on the
    EKM. If unset, this defaults to MANUAL.

    Values:
      KEY_MANAGEMENT_MODE_UNSPECIFIED: Not specified.
      MANUAL: EKM-side key management operations on CryptoKeys created with
        this EkmConnection must be initiated from the EKM directly and cannot
        be performed from Cloud KMS. This means that: * When creating a
        CryptoKeyVersion associated with this EkmConnection, the caller must
        supply the key path of pre-existing external key material that will be
        linked to the CryptoKeyVersion. * Destruction of external key material
        cannot be requested via the Cloud KMS API and must be performed
        directly in the EKM. * Automatic rotation of key material is not
        supported.
      CLOUD_KMS: All CryptoKeys created with this EkmConnection use EKM-side
        key management operations initiated from Cloud KMS. This means that: *
        When a CryptoKeyVersion associated with this EkmConnection is created,
        the EKM automatically generates new key material and a new key path.
        The caller cannot supply the key path of pre-existing external key
        material. * Destruction of external key material associated with this
        EkmConnection can be requested by calling DestroyCryptoKeyVersion. *
        Automatic rotation of key material is supported.
    """
    KEY_MANAGEMENT_MODE_UNSPECIFIED = 0
    MANUAL = 1
    CLOUD_KMS = 2