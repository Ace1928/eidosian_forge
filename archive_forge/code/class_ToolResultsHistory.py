from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ToolResultsHistory(_messages.Message):
    """Represents a tool results history resource.

  Fields:
    historyId: Required. A tool results history ID.
    projectId: Required. The cloud project that owns the tool results history.
  """
    historyId = _messages.StringField(1)
    projectId = _messages.StringField(2)