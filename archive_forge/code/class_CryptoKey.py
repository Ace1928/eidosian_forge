from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CryptoKey(_messages.Message):
    """A CryptoKey represents a logical key that can be used for cryptographic
  operations.  A CryptoKey is made up of one or more versions, which represent
  the actual key material used in cryptographic operations.

  Enums:
    PurposeValueValuesEnum: The immutable purpose of this CryptoKey.
      Currently, the only acceptable purpose is ENCRYPT_DECRYPT.

  Messages:
    LabelsValue: Labels with user defined metadata.

  Fields:
    createTime: Output only. The time at which this CryptoKey was created.
    labels: Labels with user defined metadata.
    name: Output only. The resource name for this CryptoKey in the format
      `projects/*/locations/*/keyRings/*/cryptoKeys/*`.
    nextRotationTime: At next_rotation_time, the Key Management Service will
      automatically:  1. Create a new version of this CryptoKey. 2. Mark the
      new version as primary.  Key rotations performed manually via
      CreateCryptoKeyVersion and UpdateCryptoKeyPrimaryVersion do not affect
      next_rotation_time.
    primary: Output only. A copy of the "primary" CryptoKeyVersion that will
      be used by Encrypt when this CryptoKey is given in EncryptRequest.name.
      The CryptoKey's primary version can be updated via
      UpdateCryptoKeyPrimaryVersion.
    purpose: The immutable purpose of this CryptoKey. Currently, the only
      acceptable purpose is ENCRYPT_DECRYPT.
    rotationPeriod: next_rotation_time will be advanced by this period when
      the service automatically rotates a key. Must be at least one day.  If
      rotation_period is set, next_rotation_time must also be set.
  """

    class PurposeValueValuesEnum(_messages.Enum):
        """The immutable purpose of this CryptoKey. Currently, the only acceptable
    purpose is ENCRYPT_DECRYPT.

    Values:
      CRYPTO_KEY_PURPOSE_UNSPECIFIED: Not specified.
      ENCRYPT_DECRYPT: CryptoKeys with this purpose may be used with Encrypt
        and Decrypt.
    """
        CRYPTO_KEY_PURPOSE_UNSPECIFIED = 0
        ENCRYPT_DECRYPT = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels with user defined metadata.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)
    name = _messages.StringField(3)
    nextRotationTime = _messages.StringField(4)
    primary = _messages.MessageField('CryptoKeyVersion', 5)
    purpose = _messages.EnumField('PurposeValueValuesEnum', 6)
    rotationPeriod = _messages.StringField(7)