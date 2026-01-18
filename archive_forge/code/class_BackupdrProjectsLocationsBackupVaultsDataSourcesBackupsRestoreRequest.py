from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsRestoreRequest(_messages.Message):
    """A BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsRestoreRequest
  object.

  Fields:
    name: Required. The resource name of the Backup instance, in the format
      `projects/*/locations/*/backupVaults/*/dataSources/*/backups/`.
    restoreBackupRequest: A RestoreBackupRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    restoreBackupRequest = _messages.MessageField('RestoreBackupRequest', 2)