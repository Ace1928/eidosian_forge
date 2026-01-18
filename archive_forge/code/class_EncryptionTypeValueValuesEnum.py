from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptionTypeValueValuesEnum(_messages.Enum):
    """Required. The encryption type of the restored database.

    Values:
      ENCRYPTION_TYPE_UNSPECIFIED: Unspecified. Do not use.
      USE_CONFIG_DEFAULT_OR_BACKUP_ENCRYPTION: This is the default option when
        encryption_config is not specified.
      GOOGLE_DEFAULT_ENCRYPTION: Use Google default encryption.
      CUSTOMER_MANAGED_ENCRYPTION: Use customer managed encryption. If
        specified, `kms_key_name` must must contain a valid Cloud KMS key.
    """
    ENCRYPTION_TYPE_UNSPECIFIED = 0
    USE_CONFIG_DEFAULT_OR_BACKUP_ENCRYPTION = 1
    GOOGLE_DEFAULT_ENCRYPTION = 2
    CUSTOMER_MANAGED_ENCRYPTION = 3