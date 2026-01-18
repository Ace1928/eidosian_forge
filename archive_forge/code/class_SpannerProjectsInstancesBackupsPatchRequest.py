from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesBackupsPatchRequest(_messages.Message):
    """A SpannerProjectsInstancesBackupsPatchRequest object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    name: Output only for the CreateBackup operation. Required for the
      UpdateBackup operation. A globally unique identifier for the backup
      which cannot be changed. Values are of the form
      `projects//instances//backups/a-z*[a-z0-9]` The final segment of the
      name must be between 2 and 60 characters in length. The backup is stored
      in the location(s) specified in the instance configuration of the
      instance containing the backup, identified by the prefix of the backup
      name of the form `projects//instances/`.
    updateMask: Required. A mask specifying which fields (e.g. `expire_time`)
      in the Backup resource should be updated. This mask is relative to the
      Backup resource, not to the request message. The field mask must always
      be specified; this prevents any future fields from being erased
      accidentally by clients that do not know about them.
  """
    backup = _messages.MessageField('Backup', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)