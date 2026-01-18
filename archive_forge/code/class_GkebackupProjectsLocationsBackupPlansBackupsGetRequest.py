from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsBackupPlansBackupsGetRequest(_messages.Message):
    """A GkebackupProjectsLocationsBackupPlansBackupsGetRequest object.

  Fields:
    name: Required. Full name of the Backup resource. Format:
      `projects/*/locations/*/backupPlans/*/backups/*`
  """
    name = _messages.StringField(1, required=True)