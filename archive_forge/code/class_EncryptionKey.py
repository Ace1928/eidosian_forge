from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptionKey(_messages.Message):
    """Encryption Key value.

  Enums:
    TypeValueValuesEnum: Type.

  Fields:
    kmsKeyName: The [KMS key name] with which the content of the Operation is
      encrypted. The expected format:
      `projects/*/locations/*/keyRings/*/cryptoKeys/*`. Will be empty string
      if google managed.
    type: Type.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type.

    Values:
      TYPE_UNSPECIFIED: Value type is not specified.
      GOOGLE_MANAGED: Google Managed.
      CUSTOMER_MANAGED: Customer Managed.
    """
        TYPE_UNSPECIFIED = 0
        GOOGLE_MANAGED = 1
        CUSTOMER_MANAGED = 2
    kmsKeyName = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)