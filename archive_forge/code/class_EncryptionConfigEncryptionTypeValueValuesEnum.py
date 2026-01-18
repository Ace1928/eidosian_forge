from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptionConfigEncryptionTypeValueValuesEnum(_messages.Enum):
    """Required. The encryption type of the backup.

    Values:
      ENCRYPTION_TYPE_UNSPECIFIED: Unspecified. Do not use.
      USE_DATABASE_ENCRYPTION: Use the same encryption configuration as the
        database. This is the default option when encryption_config is empty.
        For example, if the database is using `Customer_Managed_Encryption`,
        the backup will be using the same Cloud KMS key as the database.
      GOOGLE_DEFAULT_ENCRYPTION: Use Google default encryption.
      CUSTOMER_MANAGED_ENCRYPTION: Use customer managed encryption. If
        specified, `kms_key_name` must contain a valid Cloud KMS key.
    """
    ENCRYPTION_TYPE_UNSPECIFIED = 0
    USE_DATABASE_ENCRYPTION = 1
    GOOGLE_DEFAULT_ENCRYPTION = 2
    CUSTOMER_MANAGED_ENCRYPTION = 3