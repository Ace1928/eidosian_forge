from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsBackupPlansBackupsVolumeBackupsGetRequest(_messages.Message):
    """A GkebackupProjectsLocationsBackupPlansBackupsVolumeBackupsGetRequest
  object.

  Fields:
    name: Required. Full name of the VolumeBackup resource. Format:
      `projects/*/locations/*/backupPlans/*/backups/*/volumeBackups/*`
  """
    name = _messages.StringField(1, required=True)