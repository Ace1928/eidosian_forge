from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommitConversionWorkspaceRequest(_messages.Message):
    """Request message for 'CommitConversionWorkspace' request.

  Fields:
    commitName: Optional. Optional name of the commit.
  """
    commitName = _messages.StringField(1)