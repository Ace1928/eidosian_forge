from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlServerBackups(_messages.Message):
    """Specifies the backup details in Cloud Storage for homogeneous migration
  to Cloud SQL for SQL Server.

  Fields:
    gcsBucket: Required. The Cloud Storage bucket that stores backups for all
      replicated databases.
    gcsPrefix: Optional. Cloud Storage path inside the bucket that stores
      backups.
  """
    gcsBucket = _messages.StringField(1)
    gcsPrefix = _messages.StringField(2)