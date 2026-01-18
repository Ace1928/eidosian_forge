from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ModifyWorkspaceRequest(_messages.Message):
    """Request for ModifyWorkspace.

  Fields:
    actions: An ordered sequence of actions to perform in the workspace.  May
      not be empty.
    currentSnapshotId: If non-empty, current_snapshot_id must refer to the
      most recent update to the workspace, or ABORTED is returned.
    workspaceId: The ID of the workspace.
  """
    actions = _messages.MessageField('Action', 1, repeated=True)
    currentSnapshotId = _messages.StringField(2)
    workspaceId = _messages.MessageField('CloudWorkspaceId', 3)