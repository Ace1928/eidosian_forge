from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSnapshotsListRequest(_messages.Message):
    """A PubsubProjectsSnapshotsListRequest object.

  Fields:
    pageSize: Optional. Maximum number of snapshots to return.
    pageToken: Optional. The value returned by the last
      `ListSnapshotsResponse`; indicates that this is a continuation of a
      prior `ListSnapshots` call, and that the system should return the next
      page of data.
    project: Required. The name of the project in which to list snapshots.
      Format is `projects/{project-id}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    project = _messages.StringField(3, required=True)