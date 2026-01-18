from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSnapshotsGetRequest(_messages.Message):
    """A PubsubProjectsSnapshotsGetRequest object.

  Fields:
    snapshot: Required. The name of the snapshot to get. Format is
      `projects/{project}/snapshots/{snap}`.
  """
    snapshot = _messages.StringField(1, required=True)