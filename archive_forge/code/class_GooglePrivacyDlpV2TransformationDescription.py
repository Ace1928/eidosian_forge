from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TransformationDescription(_messages.Message):
    """A flattened description of a `PrimitiveTransformation` or
  `RecordSuppression`.

  Enums:
    TypeValueValuesEnum: The transformation type.

  Fields:
    condition: A human-readable string representation of the `RecordCondition`
      corresponding to this transformation. Set if a `RecordCondition` was
      used to determine whether or not to apply this transformation. Examples:
      * (age_field > 85) * (age_field <= 18) * (zip_field exists) * (zip_field
      == 01234) && (city_field != "Springville") * (zip_field == 01234) &&
      (age_field <= 18) && (city_field exists)
    description: A description of the transformation. This is empty for a
      RECORD_SUPPRESSION, or is the output of calling toString() on the
      `PrimitiveTransformation` protocol buffer message for any other type of
      transformation.
    infoType: Set if the transformation was limited to a specific `InfoType`.
    type: The transformation type.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The transformation type.

    Values:
      TRANSFORMATION_TYPE_UNSPECIFIED: Unused
      RECORD_SUPPRESSION: Record suppression
      REPLACE_VALUE: Replace value
      REPLACE_DICTIONARY: Replace value using a dictionary.
      REDACT: Redact
      CHARACTER_MASK: Character mask
      CRYPTO_REPLACE_FFX_FPE: FFX-FPE
      FIXED_SIZE_BUCKETING: Fixed size bucketing
      BUCKETING: Bucketing
      REPLACE_WITH_INFO_TYPE: Replace with info type
      TIME_PART: Time part
      CRYPTO_HASH: Crypto hash
      DATE_SHIFT: Date shift
      CRYPTO_DETERMINISTIC_CONFIG: Deterministic crypto
      REDACT_IMAGE: Redact image
    """
        TRANSFORMATION_TYPE_UNSPECIFIED = 0
        RECORD_SUPPRESSION = 1
        REPLACE_VALUE = 2
        REPLACE_DICTIONARY = 3
        REDACT = 4
        CHARACTER_MASK = 5
        CRYPTO_REPLACE_FFX_FPE = 6
        FIXED_SIZE_BUCKETING = 7
        BUCKETING = 8
        REPLACE_WITH_INFO_TYPE = 9
        TIME_PART = 10
        CRYPTO_HASH = 11
        DATE_SHIFT = 12
        CRYPTO_DETERMINISTIC_CONFIG = 13
        REDACT_IMAGE = 14
    condition = _messages.StringField(1)
    description = _messages.StringField(2)
    infoType = _messages.MessageField('GooglePrivacyDlpV2InfoType', 3)
    type = _messages.EnumField('TypeValueValuesEnum', 4)