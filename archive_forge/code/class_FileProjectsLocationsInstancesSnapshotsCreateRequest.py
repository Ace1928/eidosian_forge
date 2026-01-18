from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsInstancesSnapshotsCreateRequest(_messages.Message):
    """A FileProjectsLocationsInstancesSnapshotsCreateRequest object.

  Fields:
    parent: Required. The Filestore Instance to create the snapshots of, in
      the format
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    snapshot: A Snapshot resource to be passed as the request body.
    snapshotId: Required. The ID to use for the snapshot. The ID must be
      unique within the specified instance. This value must start with a
      lowercase letter followed by up to 62 lowercase letters, numbers, or
      hyphens, and cannot end with a hyphen.
  """
    parent = _messages.StringField(1, required=True)
    snapshot = _messages.MessageField('Snapshot', 2)
    snapshotId = _messages.StringField(3)