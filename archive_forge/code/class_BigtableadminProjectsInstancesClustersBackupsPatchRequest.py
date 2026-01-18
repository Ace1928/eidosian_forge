from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersBackupsPatchRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersBackupsPatchRequest object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    name: A globally unique identifier for the backup which cannot be changed.
      Values are of the form
      `projects/{project}/instances/{instance}/clusters/{cluster}/ backups/_a-
      zA-Z0-9*` The final segment of the name must be between 1 and 50
      characters in length. The backup is stored in the cluster identified by
      the prefix of the backup name of the form
      `projects/{project}/instances/{instance}/clusters/{cluster}`.
    updateMask: Required. A mask specifying which fields (e.g. `expire_time`)
      in the Backup resource should be updated. This mask is relative to the
      Backup resource, not to the request message. The field mask must always
      be specified; this prevents any future fields from being erased
      accidentally by clients that do not know about them.
  """
    backup = _messages.MessageField('Backup', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)