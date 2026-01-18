from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptionInfo(_messages.Message):
    """Encryption information for a Cloud Spanner database or backup.

  Enums:
    EncryptionTypeValueValuesEnum: Output only. The type of encryption.

  Fields:
    encryptionStatus: Output only. If present, the status of a recent
      encrypt/decrypt call on underlying data for this database or backup.
      Regardless of status, data is always encrypted at rest.
    encryptionType: Output only. The type of encryption.
    kmsKeyVersion: Output only. A Cloud KMS key version that is being used to
      protect the database or backup.
  """

    class EncryptionTypeValueValuesEnum(_messages.Enum):
        """Output only. The type of encryption.

    Values:
      TYPE_UNSPECIFIED: Encryption type was not specified, though data at rest
        remains encrypted.
      GOOGLE_DEFAULT_ENCRYPTION: The data is encrypted at rest with a key that
        is fully managed by Google. No key version or status will be
        populated. This is the default state.
      CUSTOMER_MANAGED_ENCRYPTION: The data is encrypted at rest with a key
        that is managed by the customer. The active version of the key.
        `kms_key_version` will be populated, and `encryption_status` may be
        populated.
    """
        TYPE_UNSPECIFIED = 0
        GOOGLE_DEFAULT_ENCRYPTION = 1
        CUSTOMER_MANAGED_ENCRYPTION = 2
    encryptionStatus = _messages.MessageField('Status', 1)
    encryptionType = _messages.EnumField('EncryptionTypeValueValuesEnum', 2)
    kmsKeyVersion = _messages.StringField(3)