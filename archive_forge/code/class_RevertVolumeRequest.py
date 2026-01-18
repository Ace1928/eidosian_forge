from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RevertVolumeRequest(_messages.Message):
    """RevertVolumeRequest reverts the given volume to the specified snapshot.

  Fields:
    snapshotId: Required. The snapshot resource ID, in the format 'my-
      snapshot', where the specified ID is the {snapshot_id} of the fully
      qualified name like projects/{project_id}/locations/{location_id}/volume
      s/{volume_id}/snapshots/{snapshot_id}
  """
    snapshotId = _messages.StringField(1)