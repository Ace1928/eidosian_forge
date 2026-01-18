from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsSnapshotsPatchRequest(_messages.Message):
    """A FileProjectsLocationsSnapshotsPatchRequest object.

  Fields:
    name: Output only. The resource name of the snapshot, in the format
      `projects/{project_id}/locations/{location_id}/snapshots/{snapshot_id}`.
    snapshot: A Snapshot resource to be passed as the request body.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field.
  """
    name = _messages.StringField(1, required=True)
    snapshot = _messages.MessageField('Snapshot', 2)
    updateMask = _messages.StringField(3)