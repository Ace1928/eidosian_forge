from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsBackupPlansBackupsCreateRequest(_messages.Message):
    """A GkebackupProjectsLocationsBackupPlansBackupsCreateRequest object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    backupId: Optional. The client-provided short name for the Backup
      resource. This name must: - be between 1 and 63 characters long
      (inclusive) - consist of only lower-case ASCII letters, numbers, and
      dashes - start with a lower-case letter - end with a lower-case letter
      or number - be unique within the set of Backups in this BackupPlan
    parent: Required. The BackupPlan within which to create the Backup.
      Format: `projects/*/locations/*/backupPlans/*`
  """
    backup = _messages.MessageField('Backup', 1)
    backupId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)