from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SaveSnapshotResponse(_messages.Message):
    """Response to SaveSnapshotRequest.

  Fields:
    snapshotPath: The fully-resolved Cloud Storage path of the created
      snapshot, e.g.: "gs://my-
      bucket/snapshots/project_location_environment_timestamp". This field is
      populated only if the snapshot creation was successful.
  """
    snapshotPath = _messages.StringField(1)