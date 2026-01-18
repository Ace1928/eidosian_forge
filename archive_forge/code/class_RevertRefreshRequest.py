from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RevertRefreshRequest(_messages.Message):
    """Request for RevertRefresh.

  Fields:
    workspaceId: The ID of the workspace.
  """
    workspaceId = _messages.MessageField('CloudWorkspaceId', 1)