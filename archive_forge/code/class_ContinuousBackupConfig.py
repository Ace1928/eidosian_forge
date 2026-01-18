from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContinuousBackupConfig(_messages.Message):
    """ContinuousBackupConfig describes the continuous backups recovery
  configurations of a cluster.

  Fields:
    enabled: Whether ContinuousBackup is enabled.
    encryptionConfig: The encryption config can be specified to encrypt the
      backups with a customer-managed encryption key (CMEK). When this field
      is not specified, the backup will then use default encryption scheme to
      protect the user data.
    enforcedRetention: If true, backups created by this config would have
      `enforced_retention` set and cannot be deleted unless they expire (or as
      part of project deletion).
    recoveryWindowDays: The number of days that are eligible to restore from
      using PITR. To support the entire recovery window, backups and logs are
      retained for one day more than the recovery window. If not set, defaults
      to 14 days.
  """
    enabled = _messages.BooleanField(1)
    encryptionConfig = _messages.MessageField('EncryptionConfig', 2)
    enforcedRetention = _messages.BooleanField(3)
    recoveryWindowDays = _messages.IntegerField(4, variant=_messages.Variant.INT32)