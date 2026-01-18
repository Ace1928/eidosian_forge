from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsBackupsPatchRequest(_messages.Message):
    """A FileProjectsLocationsBackupsPatchRequest object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    name: Output only. The resource name of the backup, in the format
      `projects/{project_number}/locations/{location_id}/backups/{backup_id}`.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field.
  """
    backup = _messages.MessageField('Backup', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)