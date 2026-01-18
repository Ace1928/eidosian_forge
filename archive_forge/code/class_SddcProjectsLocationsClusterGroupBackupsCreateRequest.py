from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupBackupsCreateRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupBackupsCreateRequest object.

  Fields:
    clusterGroupBackup: A ClusterGroupBackup resource to be passed as the
      request body.
    clusterGroupBackupId: Required. The user-provided ID of the
      `ClusterGroupBackup` to create. This ID must be unique among
      `ClusterGroupBackup` objects within the parent and becomes the final
      token in the name URI.
    parent: Required. The location (region) and project where the new
      `ClusterGroupBackup` is created. For example, projects/PROJECT-NUMBER
      /locations/us-central1
    requestId: UUID of this invocation for idempotent operation.
  """
    clusterGroupBackup = _messages.MessageField('ClusterGroupBackup', 1)
    clusterGroupBackupId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)