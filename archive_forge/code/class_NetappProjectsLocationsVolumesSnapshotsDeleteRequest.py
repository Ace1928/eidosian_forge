from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsVolumesSnapshotsDeleteRequest(_messages.Message):
    """A NetappProjectsLocationsVolumesSnapshotsDeleteRequest object.

  Fields:
    name: Required. The snapshot resource name, in the format
      `projects/*/locations/*/volumes/*/snapshots/{snapshot_id}`
  """
    name = _messages.StringField(1, required=True)