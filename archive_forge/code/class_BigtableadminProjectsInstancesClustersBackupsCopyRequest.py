from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersBackupsCopyRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersBackupsCopyRequest object.

  Fields:
    copyBackupRequest: A CopyBackupRequest resource to be passed as the
      request body.
    parent: Required. The name of the destination cluster that will contain
      the backup copy. The cluster must already exists. Values are of the
      form: `projects/{project}/instances/{instance}/clusters/{cluster}`.
  """
    copyBackupRequest = _messages.MessageField('CopyBackupRequest', 1)
    parent = _messages.StringField(2, required=True)