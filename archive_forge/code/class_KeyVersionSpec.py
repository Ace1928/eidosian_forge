from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyVersionSpec(_messages.Message):
    """A Cloud KMS key configuration that a CertificateAuthority will use.

  Enums:
    AlgorithmValueValuesEnum: The algorithm to use for creating a managed
      Cloud KMS key for a for a simplified experience. All managed keys will
      be have their ProtectionLevel as `HSM`.

  Fields:
    algorithm: The algorithm to use for creating a managed Cloud KMS key for a
      for a simplified experience. All managed keys will be have their
      ProtectionLevel as `HSM`.
    cloudKmsKeyVersion: The resource name for an existing Cloud KMS
      CryptoKeyVersion in the format
      `projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*`.
      This option enables full flexibility in the key's capabilities and
      properties.
  """

    class AlgorithmValueValuesEnum(_messages.Enum):
        """The algorithm to use for creating a managed Cloud KMS key for a for a
    simplified experience. All managed keys will be have their ProtectionLevel
    as `HSM`.

    Values:
      SIGN_HASH_ALGORITHM_UNSPECIFIED: Not specified.
      RSA_PSS_2048_SHA256: maps to
        CryptoKeyVersionAlgorithm.RSA_SIGN_PSS_2048_SHA256
      RSA_PSS_3072_SHA256: maps to CryptoKeyVersionAlgorithm.
        RSA_SIGN_PSS_3072_SHA256
      RSA_PSS_4096_SHA256: maps to
        CryptoKeyVersionAlgorithm.RSA_SIGN_PSS_4096_SHA256
      RSA_PKCS1_2048_SHA256: maps to
        CryptoKeyVersionAlgorithm.RSA_SIGN_PKCS1_2048_SHA256
      RSA_PKCS1_3072_SHA256: maps to
        CryptoKeyVersionAlgorithm.RSA_SIGN_PKCS1_3072_SHA256
      RSA_PKCS1_4096_SHA256: maps to
        CryptoKeyVersionAlgorithm.RSA_SIGN_PKCS1_4096_SHA256
      EC_P256_SHA256: maps to CryptoKeyVersionAlgorithm.EC_SIGN_P256_SHA256
      EC_P384_SHA384: maps to CryptoKeyVersionAlgorithm.EC_SIGN_P384_SHA384
    """
        SIGN_HASH_ALGORITHM_UNSPECIFIED = 0
        RSA_PSS_2048_SHA256 = 1
        RSA_PSS_3072_SHA256 = 2
        RSA_PSS_4096_SHA256 = 3
        RSA_PKCS1_2048_SHA256 = 4
        RSA_PKCS1_3072_SHA256 = 5
        RSA_PKCS1_4096_SHA256 = 6
        EC_P256_SHA256 = 7
        EC_P384_SHA384 = 8
    algorithm = _messages.EnumField('AlgorithmValueValuesEnum', 1)
    cloudKmsKeyVersion = _messages.StringField(2)