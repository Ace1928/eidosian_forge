from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SaveSnapshotRequest(_messages.Message):
    """Request to create a snapshot of a Cloud Composer environment.

  Fields:
    snapshotLocation: Location in a Cloud Storage where the snapshot is going
      to be stored, e.g.: "gs://my-bucket/snapshots".
  """
    snapshotLocation = _messages.StringField(1)