from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlBackupsCreateBackupRequest(_messages.Message):
    """A SqlBackupsCreateBackupRequest object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    parent: Required. The parent resource where this backup will be created.
      Format: projects/{project}
  """
    backup = _messages.MessageField('Backup', 1)
    parent = _messages.StringField(2, required=True)