from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSnapshotsDeleteRequest(_messages.Message):
    """A PubsubProjectsSnapshotsDeleteRequest object.

  Fields:
    snapshot: Required. The name of the snapshot to delete. Format is
      `projects/{project}/snapshots/{snap}`.
  """
    snapshot = _messages.StringField(1, required=True)