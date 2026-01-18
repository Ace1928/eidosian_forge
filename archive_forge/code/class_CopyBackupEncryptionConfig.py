from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CopyBackupEncryptionConfig(_messages.Message):
    """Encryption configuration for the copied backup.

  Enums:
    EncryptionTypeValueValuesEnum: Required. The encryption type of the
      backup.

  Fields:
    encryptionType: Required. The encryption type of the backup.
    kmsKeyName: Optional. The Cloud KMS key that will be used to protect the
      backup. This field should be set only when encryption_type is
      `CUSTOMER_MANAGED_ENCRYPTION`. Values are of the form
      `projects//locations//keyRings//cryptoKeys/`.
    kmsKeyNames: Optional. Specifies the KMS configuration for the one or more
      keys used to protect the backup. Values are of the form
      `projects//locations//keyRings//cryptoKeys/`. Kms keys specified can be
      in any order. The keys referenced by kms_key_names must fully cover all
      regions of the backup's instance configuration. Some examples: * For
      single region instance configs, specify a single regional location KMS
      key. * For multi-regional instance configs of type GOOGLE_MANAGED,
      either specify a multi-regional location KMS key or multiple regional
      location KMS keys that cover all regions in the instance config. * For
      an instance config of type USER_MANAGED, please specify only regional
      location KMS keys to cover each region in the instance config. Multi-
      regional location KMS keys are not supported for USER_MANAGED instance
      configs.
  """

    class EncryptionTypeValueValuesEnum(_messages.Enum):
        """Required. The encryption type of the backup.

    Values:
      ENCRYPTION_TYPE_UNSPECIFIED: Unspecified. Do not use.
      USE_CONFIG_DEFAULT_OR_BACKUP_ENCRYPTION: This is the default option for
        CopyBackup when encryption_config is not specified. For example, if
        the source backup is using `Customer_Managed_Encryption`, the backup
        will be using the same Cloud KMS key as the source backup.
      GOOGLE_DEFAULT_ENCRYPTION: Use Google default encryption.
      CUSTOMER_MANAGED_ENCRYPTION: Use customer managed encryption. If
        specified, either `kms_key_name` or `kms_key_names` must contain valid
        Cloud KMS key(s).
    """
        ENCRYPTION_TYPE_UNSPECIFIED = 0
        USE_CONFIG_DEFAULT_OR_BACKUP_ENCRYPTION = 1
        GOOGLE_DEFAULT_ENCRYPTION = 2
        CUSTOMER_MANAGED_ENCRYPTION = 3
    encryptionType = _messages.EnumField('EncryptionTypeValueValuesEnum', 1)
    kmsKeyName = _messages.StringField(2)
    kmsKeyNames = _messages.StringField(3, repeated=True)