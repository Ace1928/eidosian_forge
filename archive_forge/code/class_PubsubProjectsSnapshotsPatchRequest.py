from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSnapshotsPatchRequest(_messages.Message):
    """A PubsubProjectsSnapshotsPatchRequest object.

  Fields:
    name: Optional. The name of the snapshot.
    updateSnapshotRequest: A UpdateSnapshotRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    updateSnapshotRequest = _messages.MessageField('UpdateSnapshotRequest', 2)