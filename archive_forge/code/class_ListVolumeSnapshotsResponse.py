from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListVolumeSnapshotsResponse(_messages.Message):
    """Response message containing the list of volume snapshots.

  Fields:
    nextPageToken: A token identifying a page of results from the server.
    unreachable: Locations that could not be reached.
    volumeSnapshots: The list of snapshots.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    volumeSnapshots = _messages.MessageField('VolumeSnapshot', 3, repeated=True)