from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsBackupPlansDeleteRequest(_messages.Message):
    """A GkebackupProjectsLocationsBackupPlansDeleteRequest object.

  Fields:
    etag: Optional. If provided, this value must match the current value of
      the target BackupPlan's etag field or the request is rejected.
    name: Required. Fully qualified BackupPlan name. Format:
      `projects/*/locations/*/backupPlans/*`
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)