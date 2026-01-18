from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesSnapshotsPatchRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesSnapshotsPatchRequest object.

  Fields:
    name: Identifier. The resource name of the snapshot. Format: `projects/{pr
      oject_id}/locations/{location}/volumes/{volume_id}/snapshots/{snapshot_i
      d}`.
    snapshot: A Snapshot resource to be passed as the request body.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field.
  """
    name = _messages.StringField(1, required=True)
    snapshot = _messages.MessageField('Snapshot', 2)
    updateMask = _messages.StringField(3)