from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudWorkspaceSourceContext(_messages.Message):
    """A CloudWorkspaceSourceContext denotes a workspace at a particular
  snapshot.

  Fields:
    snapshotId: The ID of the snapshot. An empty snapshot_id refers to the
      most recent snapshot.
    workspaceId: The ID of the workspace.
  """
    snapshotId = _messages.StringField(1)
    workspaceId = _messages.MessageField('CloudWorkspaceId', 2)