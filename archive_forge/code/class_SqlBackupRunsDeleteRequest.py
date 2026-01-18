from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlBackupRunsDeleteRequest(_messages.Message):
    """A SqlBackupRunsDeleteRequest object.

  Fields:
    id: The ID of the backup run to delete. To find a backup run ID, use the
      [list](https://cloud.google.com/sql/docs/mysql/admin-
      api/rest/v1beta4/backupRuns/list) method.
    instance: Cloud SQL instance ID. This does not include the project ID.
    project: Project ID of the project that contains the instance.
  """
    id = _messages.IntegerField(1, required=True)
    instance = _messages.StringField(2, required=True)
    project = _messages.StringField(3, required=True)