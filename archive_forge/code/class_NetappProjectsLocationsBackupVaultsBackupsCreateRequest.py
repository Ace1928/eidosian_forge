from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsBackupVaultsBackupsCreateRequest(_messages.Message):
    """A NetappProjectsLocationsBackupVaultsBackupsCreateRequest object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    backupId: Required. The ID to use for the backup. The ID must be unique
      within the specified backupVault. This value must start with a lowercase
      letter followed by up to 62 lowercase letters, numbers, or hyphens, and
      cannot end with a hyphen. Values that do not match this pattern will
      trigger an INVALID_ARGUMENT error.
    parent: Required. The NetApp backupVault to create the backups of, in the
      format `projects/*/locations/*/backupVaults/{backup_vault_id}`
  """
    backup = _messages.MessageField('Backup', 1)
    backupId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)